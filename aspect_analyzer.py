#!/usr/bin/env python3
"""
Steam Review Aspect Analyzer
Processes CSV output from sentiment analysis to group and count similar aspects
"""
import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import ollama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral:instruct')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
BATCH_SIZE = int(os.getenv('ASPECT_BATCH_SIZE', '50'))  # Number of aspects to group at once

class AspectAnalyzer:
    def __init__(self, model: str = OLLAMA_MODEL, ollama_host: str = OLLAMA_HOST):
        self.model = model
        self.ollama_host = ollama_host
        
        # Configure Ollama client if host is specified
        if ollama_host != "http://localhost:11434":
            ollama.Client(host=ollama_host)
        
        # Test Ollama connection
        try:
            ollama.list()
            print(f"✓ Connected to Ollama at {ollama_host} using model: {self.model}")
        except Exception as e:
            print(f"✗ Failed to connect to Ollama at {ollama_host}: {e}")
            print("Make sure Ollama is running and the model is pulled")
            raise

    def load_csv_data(self, csv_file: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(csv_file)
            required_columns = ['positive_aspects', 'negative_aspects']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV")
            
            print(f"✓ Loaded {len(df)} reviews from {csv_file}")
            return df
            
        except Exception as e:
            print(f"✗ Error loading CSV file: {e}")
            raise

    def extract_aspects(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Extract and clean individual aspects from the CSV data"""
        positive_aspects = []
        negative_aspects = []
        
        print("Extracting aspects from reviews...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
            # Process positive aspects
            if pd.notna(row['positive_aspects']) and row['positive_aspects'].strip():
                pos_aspects = [aspect.strip().lower() for aspect in str(row['positive_aspects']).split(',')]
                pos_aspects = [aspect for aspect in pos_aspects if aspect and len(aspect) > 1]
                positive_aspects.extend(pos_aspects)
            
            # Process negative aspects
            if pd.notna(row['negative_aspects']) and row['negative_aspects'].strip():
                neg_aspects = [aspect.strip().lower() for aspect in str(row['negative_aspects']).split(',')]
                neg_aspects = [aspect for aspect in neg_aspects if aspect and len(aspect) > 1]
                negative_aspects.extend(neg_aspects)
        
        print(f"✓ Extracted {len(positive_aspects)} positive aspects and {len(negative_aspects)} negative aspects")
        return positive_aspects, negative_aspects

    def get_initial_counts(self, aspects: List[str]) -> Dict[str, int]:
        """Get initial counts of exact aspect matches"""
        return dict(Counter(aspects))

    def create_grouping_prompt(self, aspects_batch: List[str]) -> str:
        """Create a prompt for grouping similar aspects"""
        aspects_list = "\n".join([f'"{aspect}"' for aspect in aspects_batch])
        
        prompt = f"""You must group ALL of these {len(aspects_batch)} game review aspects. Every single aspect must appear in exactly one group.

ASPECTS TO GROUP:
{aspects_list}

INSTRUCTIONS:
1. Read through ALL {len(aspects_batch)} aspects carefully
2. Group similar aspects together
3. Choose ONE existing aspect as the group name (the clearest/most representative one)
4. Every aspect must be included - do not skip any
5. Copy aspect text exactly as written, including quotes if present

EXAMPLE FORMAT:
{{
    "groups": [
        {{
            "group_name": "graphics are stunning",
            "aspects": ["graphics are stunning", "visuals look amazing"]
        }},
        {{
            "group_name": "story is boring", 
            "aspects": ["story is boring"]
        }}
    ]
}}

CRITICAL REQUIREMENTS:
- Include ALL {len(aspects_batch)} aspects
- Use exact text from the list above
- Each aspect appears in exactly one group
- Group name must be one of the aspects in that group

Return ONLY the JSON, no explanations."""
        
        return prompt

    def group_aspects_batch(self, aspects_batch: List[str], max_retries: int = 3) -> Optional[List[Dict]]:
        """Group a batch of aspects using LLM with retry logic"""
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"  Retry attempt {attempt + 1}/{max_retries}...")
                    time.sleep(1)
                
                prompt = self.create_grouping_prompt(aspects_batch)
                
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=False,
                    options={
                        'temperature': 0.05,  # Very low for consistency
                        'top_p': 0.9,
                        'max_tokens': 6000,
                        'num_ctx': 16384,  # Large context
                        'repeat_penalty': 1.05,
                    }
                )
                
                response_text = response['response'].strip()
                
                # Extract JSON more robustly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not json_match:
                    if attempt < max_retries - 1:
                        print(f"    No JSON found in response")
                    continue
                
                json_str = json_match.group(0)
                
                # Clean common JSON issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control chars
                
                result = json.loads(json_str)
                
                if 'groups' not in result or not isinstance(result['groups'], list):
                    if attempt < max_retries - 1:
                        print(f"    Invalid JSON structure")
                    continue
                
                # Enhanced validation with detailed feedback
                all_found_aspects = []
                valid_groups = []
                
                for i, group in enumerate(result['groups']):
                    if not isinstance(group, dict):
                        continue
                    if 'group_name' not in group or 'aspects' not in group:
                        continue
                    if not isinstance(group['aspects'], list):
                        continue
                    
                    # Clean aspect text (remove extra quotes, whitespace)
                    cleaned_aspects = []
                    for aspect in group['aspects']:
                        cleaned = aspect.strip().strip('"\'')
                        cleaned_aspects.append(cleaned)
                    
                    group['aspects'] = cleaned_aspects
                    group_name = group['group_name'].strip().strip('"\'')
                    group['group_name'] = group_name
                    
                    # Validate group_name is one of the aspects
                    if group_name not in group['aspects']:
                        if attempt < max_retries - 1:
                            print(f"    Group {i+1}: '{group_name}' not in its aspects")
                        continue
                    
                    all_found_aspects.extend(group['aspects'])
                    valid_groups.append(group)
                
                # Detailed validation
                original_set = set(aspects_batch)
                found_set = set(all_found_aspects)
                
                missing = original_set - found_set
                extra = found_set - original_set
                duplicates = len(all_found_aspects) - len(found_set)
                
                if len(missing) == 0 and len(extra) == 0 and duplicates == 0 and len(valid_groups) > 0:
                    return valid_groups
                
                # Detailed error reporting
                if attempt < max_retries - 1:
                    issues = []
                    if missing: 
                        issues.append(f"Missing: {len(missing)}")
                        if len(missing) <= 3:  # Show a few examples
                            issues.append(f"Examples: {list(missing)[:3]}")
                    if extra: 
                        issues.append(f"Extra: {len(extra)}")
                        if len(extra) <= 3:
                            issues.append(f"Examples: {list(extra)[:3]}")
                    if duplicates: 
                        issues.append(f"Duplicates: {duplicates}")
                    
                    print(f"    Validation failed - {', '.join(issues)}")
                    
                    # Try to fix missing aspects by adding them as individual groups
                    if len(missing) <= 3 and len(extra) == 0 and duplicates == 0:
                        print(f"    Attempting to fix by adding missing aspects...")
                        for missing_aspect in missing:
                            valid_groups.append({
                                "group_name": missing_aspect,
                                "aspects": [missing_aspect]
                            })
                        return valid_groups
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"    JSON parse error: {str(e)[:100]}...")
                continue
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    Error: {str(e)[:100]}...")
                continue
        
        print(f"  Failed to group batch after {max_retries} attempts")
        print(f"  Batch size: {len(aspects_batch)}")
        print(f"  Sample aspects: {aspects_batch[:3]}")
        
        # Return individual groups as fallback
        return [{"group_name": aspect, "aspects": [aspect]} for aspect in aspects_batch]

    def group_similar_aspects(self, aspect_counts: Dict[str, int], aspect_type: str) -> Dict[str, Dict]:
        """Group similar aspects using LLM processing"""
        print(f"\nGrouping similar {aspect_type} aspects...")
        
        # Sort aspects by frequency (most common first) for better grouping
        sorted_aspects = sorted(aspect_counts.keys(), key=lambda x: aspect_counts[x], reverse=True)
        
        # Process in batches
        all_groups = []
        total_batches = (len(sorted_aspects) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in tqdm(range(0, len(sorted_aspects), BATCH_SIZE), 
                     desc=f"Grouping {aspect_type} aspects", total=total_batches):
            batch = sorted_aspects[i:i + BATCH_SIZE]
            groups = self.group_aspects_batch(batch)
            if groups:
                all_groups.extend(groups)
            
            # Small delay between batches
            time.sleep(0.1)
        
        # Consolidate groups and calculate counts
        final_groups = {}
        for group in all_groups:
            group_name = group['group_name']
            total_count = sum(aspect_counts.get(aspect, 0) for aspect in group['aspects'])
            
            final_groups[group_name] = {
                'count': total_count,
                'aspects': group['aspects'],
                'aspect_counts': {aspect: aspect_counts.get(aspect, 0) for aspect in group['aspects']}
            }
        
        # Sort by total count (descending)
        final_groups = dict(sorted(final_groups.items(), key=lambda x: x[1]['count'], reverse=True))
        
        return final_groups

    def merge_similar_groups(self, grouped_aspects: Dict[str, Dict], aspect_type: str) -> Dict[str, Dict]:
        """Perform a second pass to merge groups that might be similar"""
        print(f"\nPerforming second pass to merge similar {aspect_type} groups...")
        
        group_names = list(grouped_aspects.keys())
        if len(group_names) <= 1:
            return grouped_aspects
        
        # Create prompt to identify similar groups
        groups_list = "\n".join([f"{i+1}. {name} (count: {grouped_aspects[name]['count']})" 
                                for i, name in enumerate(group_names)])
        
        prompt = f"""Task: Identify groups that should be merged because they refer to the same concept.

GROUPS TO REVIEW:
{groups_list}

RULES:
1. Only merge groups about the SAME concept
2. Be conservative - when unsure, keep separate
3. Use one of the existing group names as the merge target
4. Don't create new names

REQUIRED JSON FORMAT:
{{
    "merges": [
        {{
            "merge_into": "existing group name to keep",
            "groups_to_merge": ["group1", "group2"]
        }}
    ]
}}

If no merges needed: {{"merges": []}}

Return ONLY valid JSON, no other text."""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 0.05,  # Very low temperature for consistency
                    'max_tokens': 2000,
                    'num_ctx': 4096
                }
            )
            
            response_text = response['response'].strip()
            
            # Robust JSON extraction
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                result = json.loads(json_str)
                merges = result.get('merges', [])
                
                # Apply merges with validation
                applied_merges = 0
                for merge in merges:
                    if not isinstance(merge, dict):
                        continue
                    if 'merge_into' not in merge or 'groups_to_merge' not in merge:
                        continue
                        
                    target_name = merge['merge_into']
                    groups_to_merge = merge['groups_to_merge']
                    
                    if not isinstance(groups_to_merge, list):
                        continue
                    
                    if target_name in groups_to_merge and target_name in grouped_aspects:
                        # Merge all other groups into the target
                        target_data = grouped_aspects[target_name]
                        
                        for group_name in groups_to_merge:
                            if group_name != target_name and group_name in grouped_aspects:
                                source_data = grouped_aspects[group_name]
                                
                                # Merge the data
                                target_data['count'] += source_data['count']
                                target_data['aspects'].extend(source_data['aspects'])
                                target_data['aspect_counts'].update(source_data['aspect_counts'])
                                
                                # Remove the merged group
                                del grouped_aspects[group_name]
                                applied_merges += 1
                
                # Re-sort by count
                grouped_aspects = dict(sorted(grouped_aspects.items(), 
                                            key=lambda x: x[1]['count'], reverse=True))
                
                print(f"✓ Applied {applied_merges} group merges")
                
        except Exception as e:
            print(f"⚠ Warning: Could not perform group merging: {e}")
        
        return grouped_aspects

    def save_results(self, positive_groups: Dict, negative_groups: Dict, output_file: str):
        """Save grouped aspects to CSV file"""
        results = []
        
        # Add positive aspects
        for group_name, data in positive_groups.items():
            results.append({
                'aspect_type': 'positive',
                'group_name': group_name,
                'total_count': data['count'],
                'individual_aspects': ', '.join(data['aspects']),
                'aspect_breakdown': ', '.join([f"{aspect}({count})" 
                                             for aspect, count in data['aspect_counts'].items()])
            })
        
        # Add negative aspects
        for group_name, data in negative_groups.items():
            results.append({
                'aspect_type': 'negative',
                'group_name': group_name,
                'total_count': data['count'],
                'individual_aspects': ', '.join(data['aspects']),
                'aspect_breakdown': ', '.join([f"{aspect}({count})" 
                                             for aspect, count in data['aspect_counts'].items()])
            })
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to CSV
        fieldnames = ['aspect_type', 'group_name', 'total_count', 'individual_aspects', 'aspect_breakdown']
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Saved grouped aspects to {output_file}")

    def print_summary(self, positive_groups: Dict, negative_groups: Dict):
        """Print a summary of the grouped aspects"""
        print("\n" + "="*60)
        print("📊 ASPECT GROUPING SUMMARY")
        print("="*60)
        
        print(f"\n🟢 TOP POSITIVE ASPECTS:")
        print("-" * 30)
        for i, (group_name, data) in enumerate(list(positive_groups.items())[:10], 1):
            aspects_preview = ', '.join(data['aspects'][:3])
            if len(data['aspects']) > 3:
                aspects_preview += f" (+{len(data['aspects'])-3} more)"
            print(f"{i:2d}. {group_name:<20} ({data['count']:3d}) - {aspects_preview}")
        
        print(f"\n🔴 TOP NEGATIVE ASPECTS:")
        print("-" * 30)
        for i, (group_name, data) in enumerate(list(negative_groups.items())[:10], 1):
            aspects_preview = ', '.join(data['aspects'][:3])
            if len(data['aspects']) > 3:
                aspects_preview += f" (+{len(data['aspects'])-3} more)"
            print(f"{i:2d}. {group_name:<20} ({data['count']:3d}) - {aspects_preview}")
        
        print(f"\n📈 STATISTICS:")
        print("-" * 15)
        print(f"Total positive groups: {len(positive_groups)}")
        print(f"Total negative groups: {len(negative_groups)}")
        print(f"Total positive mentions: {sum(data['count'] for data in positive_groups.values())}")
        print(f"Total negative mentions: {sum(data['count'] for data in negative_groups.values())}")

def main():
    parser = argparse.ArgumentParser(description='Group and analyze aspects from Steam review sentiment analysis')
    parser.add_argument('csv_file', type=str, help='Path to CSV file from sentiment analysis')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generated)')
    parser.add_argument('--model', type=str, default=OLLAMA_MODEL,
                       help=f'Ollama model to use (default: {OLLAMA_MODEL})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size for aspect grouping (default: {BATCH_SIZE})')
    parser.add_argument('--skip-merge', action='store_true',
                       help='Skip the second pass group merging step')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"{base_name}_grouped_aspects_{timestamp}.csv"
    
    # Initialize analyzer
    try:
        analyzer = AspectAnalyzer(args.model)
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return
    
    start_time = time.time()
    
    try:
        # Load CSV data
        print("\n" + "="*50)
        print("LOADING DATA")
        print("="*50)
        df = analyzer.load_csv_data(args.csv_file)
        
        # Extract aspects
        print("\n" + "="*50)
        print("EXTRACTING ASPECTS")
        print("="*50)
        positive_aspects, negative_aspects = analyzer.extract_aspects(df)
        
        # Get initial counts
        positive_counts = analyzer.get_initial_counts(positive_aspects)
        negative_counts = analyzer.get_initial_counts(negative_aspects)
        
        print(f"✓ Found {len(positive_counts)} unique positive aspects")
        print(f"✓ Found {len(negative_counts)} unique negative aspects")
        
        # Group positive aspects
        print("\n" + "="*50)
        print("GROUPING POSITIVE ASPECTS")
        print("="*50)
        positive_groups = analyzer.group_similar_aspects(positive_counts, "positive")
        
        # Group negative aspects
        print("\n" + "="*50)
        print("GROUPING NEGATIVE ASPECTS")
        print("="*50)
        negative_groups = analyzer.group_similar_aspects(negative_counts, "negative")
        
        # Optional second pass merging
        if not args.skip_merge:
            print("\n" + "="*50)
            print("MERGING SIMILAR GROUPS")
            print("="*50)
            positive_groups = analyzer.merge_similar_groups(positive_groups, "positive")
            negative_groups = analyzer.merge_similar_groups(negative_groups, "negative")
        
        # Save results
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        analyzer.save_results(positive_groups, negative_groups, args.output)
        
        # Print summary
        analyzer.print_summary(positive_groups, negative_groups)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total processing time: {total_time:.1f} seconds")
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        return

if __name__ == "__main__":
    main()
