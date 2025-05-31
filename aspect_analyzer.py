import argparse
import pandas as pd
import ollama
import json
import os
import re
from collections import defaultdict, Counter
from tqdm import tqdm
from colorama import init, Fore, Style
from tabulate import tabulate
import time
import sys
from typing import Dict, List, Set, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral:instruct')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# Initialize colorama for colored output
init()

class DynamicAspectGrouper:
    def __init__(self, input_csv_path, model: str = OLLAMA_MODEL, ollama_host: str = OLLAMA_HOST):
        self.input_csv_path = input_csv_path
        self.game_id = self.extract_game_id_from_filename()
        self.df = None
        self.keyword_data = defaultdict(lambda: {
            'sentiment_type': None,
            'aspect_group': None,
            'reviewer_ids': set(),
            'review_count': 0
        })
        self.aspect_groups = {
            'positive': {},
            'negative': {}
        }

        # Configure Ollama client if host is specified
        self.model = model
        self.ollama_host = ollama_host
        self.ollama_client = ollama.Client(host=ollama_host)
        
        # Test Ollama connection
        try:
            self.ollama_client.list()
            print(f"✓ Connected to Ollama at {ollama_host} using model: {self.model}")
        except Exception as e:
            print(f"✗ Failed to connect to Ollama at {ollama_host}: {e}")
            print("Make sure Ollama is running and the model is pulled")
            raise
        
    def extract_game_id_from_filename(self):
        """Extract game ID from filename pattern"""
        match = re.search(r'_(\d+)_', self.input_csv_path)
        return match.group(1) if match else '553420'
    
    def normalize_keyword(self, keyword: str) -> str:
        """Convert keyword to snake_case"""
        # Remove special characters and convert to lowercase
        keyword = re.sub(r'[^\w\s-]', '', keyword.lower())
        # Replace spaces and hyphens with underscores
        keyword = re.sub(r'[-\s]+', '_', keyword)
        # Remove leading/trailing underscores
        return keyword.strip('_')
    
    def load_data(self):
        """Load the review data from CSV"""
        print(f"{Fore.CYAN}Loading data from {self.input_csv_path}...{Style.RESET_ALL}")
        self.df = pd.read_csv(self.input_csv_path)
        print(f"{Fore.GREEN}✓ Loaded {len(self.df)} reviews{Style.RESET_ALL}")
    
    def extract_keywords_from_reviews(self):
        """Extract all keywords from reviews with their associated reviews"""
        print(f"\n{Fore.CYAN}Extracting keywords from reviews...{Style.RESET_ALL}")
        
        # Process positive keywords
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing positive keywords"):
            if pd.notna(row.get('positive_aspects')):
                aspects = str(row['positive_aspects']).strip()
                if aspects:
                    for aspect in aspects.split(','):
                        aspect = aspect.strip()
                        if aspect:
                            keyword = self.normalize_keyword(aspect)
                            self.keyword_data[keyword]['sentiment_type'] = 'positive'
                            self.keyword_data[keyword]['reviewer_ids'].add(str(row['reviewer_name']))
                            self.keyword_data[keyword]['original_keyword'] = aspect
        
        # Process negative keywords
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing negative keywords"):
            if pd.notna(row.get('negative_aspects')):
                aspects = str(row['negative_aspects']).strip()
                if aspects:
                    for aspect in aspects.split(','):
                        aspect = aspect.strip()
                        if aspect:
                            keyword = self.normalize_keyword(aspect)
                            self.keyword_data[keyword]['sentiment_type'] = 'negative'
                            self.keyword_data[keyword]['reviewer_ids'].add(str(row['reviewer_name']))
                            self.keyword_data[keyword]['original_keyword'] = aspect
        
        # Update review counts
        for keyword in self.keyword_data:
            self.keyword_data[keyword]['review_count'] = len(self.keyword_data[keyword]['reviewer_ids'])
        
        print(f"{Fore.GREEN}✓ Found {len(self.keyword_data)} unique keywords{Style.RESET_ALL}")
    
    def discover_aspect_groups(self):
        """Use Ollama to discover aspect groups for keywords"""
        print(f"\n{Fore.CYAN}Discovering aspect groups for keywords...{Style.RESET_ALL}")
        
        # Separate keywords by sentiment
        positive_keywords = {}
        negative_keywords = {}
        
        for keyword, data in self.keyword_data.items():
            original = data.get('original_keyword', keyword)
            if data['sentiment_type'] == 'positive':
                positive_keywords[original] = data['review_count']
            else:
                negative_keywords[original] = data['review_count']
        
        # Get aspect groups for positive keywords
        if positive_keywords:
            self.aspect_groups['positive'] = self.get_aspect_groups_from_ollama(
                positive_keywords, 'positive'
            )
        
        # Get aspect groups for negative keywords
        if negative_keywords:
            self.aspect_groups['negative'] = self.get_aspect_groups_from_ollama(
                negative_keywords, 'negative'
            )
        
        # Assign groups to keywords
        self.assign_groups_to_keywords()
    
    def get_aspect_groups_from_ollama(self, keywords_with_counts: Dict[str, int], sentiment_type: str) -> Dict[str, List[str]]:
        """Use Ollama to create aspect groups"""
        keywords_list = [f"{kw} ({count})" for kw, count in sorted(keywords_with_counts.items(), 
                                                                   key=lambda x: x[1], reverse=True)]
        
        prompt = f"""
        Analyze these {sentiment_type} game review keywords and group them into natural categories.
        Create groups based on what aspects of the game they relate to.
        
        Keywords (with review counts):
        {', '.join(keywords_list)}
        
        Instructions:
        1. Create logical groups based on game aspects (e.g., visual elements, gameplay mechanics, technical issues)
        2. Group names should be in snake_case (e.g., visual_presentation, gameplay_mechanics)
        3. Each keyword should belong to exactly one group
        4. Create as many groups as needed to properly categorize all keywords
        
        Return ONLY a JSON object where keys are snake_case group names and values are lists of keywords.
        Example:
        {{
            "visual_presentation": ["graphics", "art style", "visuals"],
            "gameplay_mechanics": ["gameplay", "controls", "combat"],
            "technical_performance": ["bugs", "crashes", "performance"]
        }}
        """
        
        try:
            response = self.ollama_client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )
            
            response_text = response['message']['content']
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            
            if json_match:
                groups = json.loads(json_match.group())
                
                # Normalize group names to snake_case
                normalized_groups = {}
                for group_name, keywords in groups.items():
                    normalized_name = self.normalize_keyword(group_name)
                    normalized_groups[normalized_name] = keywords
                
                return normalized_groups
            else:
                raise ValueError("No valid JSON found")
                
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Ollama grouping failed: {e}{Style.RESET_ALL}")
            return self.fallback_grouping(list(keywords_with_counts.keys()))
    
    def fallback_grouping(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Simple fallback grouping"""
        groups = defaultdict(list)
        
        # Basic keyword-based grouping
        grouping_rules = {
            'visual_elements': ['graphic', 'visual', 'art', 'aesthetic', 'design', 'animation'],
            'gameplay_mechanics': ['gameplay', 'mechanic', 'control', 'combat', 'puzzle', 'exploration'],
            'story_content': ['story', 'narrative', 'character', 'plot', 'dialogue', 'lore'],
            'technical_aspects': ['performance', 'bug', 'crash', 'optimization', 'lag', 'fps', 'glitch'],
            'audio_elements': ['music', 'sound', 'audio', 'soundtrack', 'voice'],
            'value_pricing': ['price', 'worth', 'value', 'cost', 'expensive', 'cheap', 'money'],
            'community_social': ['community', 'multiplayer', 'player', 'toxic', 'communication', 'online'],
        }
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            grouped = False
            
            for group_name, triggers in grouping_rules.items():
                if any(trigger in keyword_lower for trigger in triggers):
                    groups[group_name].append(keyword)
                    grouped = True
                    break
            
            if not grouped:
                groups['other_aspects'].append(keyword)
        
        return dict(groups)
    
    def assign_groups_to_keywords(self):
        """Assign aspect groups to keywords"""
        for keyword, data in self.keyword_data.items():
            original_keyword = data.get('original_keyword', keyword)
            sentiment = data['sentiment_type']
            
            # Find which group this keyword belongs to
            group_found = False
            if sentiment in self.aspect_groups:
                for group_name, group_keywords in self.aspect_groups[sentiment].items():
                    if original_keyword in group_keywords:
                        self.keyword_data[keyword]['aspect_group'] = group_name
                        group_found = True
                        break
            
            if not group_found:
                self.keyword_data[keyword]['aspect_group'] = 'ungrouped'
    
    def create_keyword_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with keyword-level data"""
        keyword_rows = []
        
        for keyword, data in self.keyword_data.items():
            # Generate review URLs
            review_urls = [
                f"https://steamcommunity.com/profiles/{reviewer_id}/recommended/{self.game_id}/"
                for reviewer_id in sorted(data['reviewer_ids'])
            ]
            
            keyword_rows.append({
                'keyword': keyword,
                'original_keyword': data.get('original_keyword', keyword),
                'sentiment_type': data['sentiment_type'],
                'aspect_group': data['aspect_group'],
                'review_count': data['review_count'],
                'reviewer_ids': ', '.join(sorted(data['reviewer_ids'])),
                'review_urls': ' | '.join(review_urls)
            })
        
        # Create DataFrame and sort by review count
        df = pd.DataFrame(keyword_rows)
        df = df.sort_values(['sentiment_type', 'review_count'], ascending=[True, False])
        
        return df
    
    def display_summary(self, output_df: pd.DataFrame):
        """Display keyword analysis summary"""
        print("\n" + "="*70)
        print(f"{Fore.CYAN}📊 KEYWORD ANALYSIS SUMMARY{Style.RESET_ALL}")
        print("="*70)
        
        # Separate by sentiment
        positive_df = output_df[output_df['sentiment_type'] == 'positive']
        negative_df = output_df[output_df['sentiment_type'] == 'negative']
        
        # Top positive keywords
        print(f"\n{Fore.GREEN}✅ TOP POSITIVE KEYWORDS:{Style.RESET_ALL}")
        print("-" * 70)
        print(f"{'#':<3} {'Keyword':<25} {'Group':<25} {'Reviews':<10}")
        print("-" * 70)
        
        for idx, row in positive_df.head(15).iterrows():
            print(f"{idx+1:<3} {row['keyword']:<25} {row['aspect_group']:<25} {row['review_count']:<10}")
        
        # Top negative keywords
        print(f"\n{Fore.RED}❌ TOP NEGATIVE KEYWORDS:{Style.RESET_ALL}")
        print("-" * 70)
        print(f"{'#':<3} {'Keyword':<25} {'Group':<25} {'Reviews':<10}")
        print("-" * 70)
        
        for idx, row in negative_df.head(15).iterrows():
            print(f"{idx+1:<3} {row['keyword']:<25} {row['aspect_group']:<25} {row['review_count']:<10}")
        
        # Statistics by aspect group
        print(f"\n{Fore.BLUE}📈 STATISTICS BY ASPECT GROUP:{Style.RESET_ALL}")
        print("-" * 70)
        
        group_stats = output_df.groupby(['sentiment_type', 'aspect_group']).agg({
            'keyword': 'count',
            'review_count': 'sum'
        }).rename(columns={'keyword': 'keyword_count', 'review_count': 'total_reviews'})
        
        print(group_stats.to_string())
        
        # Overall statistics
        print(f"\n{Fore.YELLOW}📊 OVERALL STATISTICS:{Style.RESET_ALL}")
        print("-" * 70)
        print(f"Total unique keywords: {len(output_df)}")
        print(f"Positive keywords: {len(positive_df)}")
        print(f"Negative keywords: {len(negative_df)}")
        print(f"Total positive mentions: {positive_df['review_count'].sum()}")
        print(f"Total negative mentions: {negative_df['review_count'].sum()}")
        
        # Most discussed aspect groups
        print(f"\n{Fore.CYAN}🏆 MOST DISCUSSED ASPECT GROUPS:{Style.RESET_ALL}")
        print("-" * 70)
        
        group_totals = output_df.groupby('aspect_group')['review_count'].sum().sort_values(ascending=False)
        for group, count in group_totals.head(10).items():
            sentiment_info = output_df[output_df['aspect_group'] == group]['sentiment_type'].value_counts()
            sentiment_str = f"[+{sentiment_info.get('positive', 0)}/-{sentiment_info.get('negative', 0)}]"
            print(f"{group:<35} {count:>5} reviews {sentiment_str}")
    
    def save_results(self, output_df: pd.DataFrame, output_path: str):
        """Save results to CSV"""
        output_df.to_csv(output_path, index=False)
        print(f"\n{Fore.GREEN}✓ Results saved to {output_path}{Style.RESET_ALL}")
        
        # Also save a summary file
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Redirect print output to file
            original_stdout = sys.stdout
            sys.stdout = f
            self.display_summary(output_df)
            sys.stdout = original_stdout
        
        print(f"{Fore.GREEN}✓ Summary saved to {summary_path}{Style.RESET_ALL}")
    
    def run(self, output_path='keyword_analysis.csv'):
        """Run the complete keyword analysis"""
        start_time = time.time()
        
        try:
            # Load data
            self.load_data()
            
            # Extract keywords from reviews
            self.extract_keywords_from_reviews()
            
            # Discover aspect groups
            self.discover_aspect_groups()
            
            # Create output DataFrame
            print(f"\n{Fore.CYAN}Creating keyword analysis DataFrame...{Style.RESET_ALL}")
            output_df = self.create_keyword_dataframe()
            
            # Display summary
            self.display_summary(output_df)
            
            # Save results
            self.save_results(output_df, output_path)
            
            # Processing time
            elapsed_time = time.time() - start_time
            print(f"\n⏱️ Total processing time: {elapsed_time:.1f} seconds")
            
        except Exception as e:
            print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Analyze Steam game reviews using local LLM')
    parser.add_argument('reviews', type=str, help='Review .csv path to analyze')
    parser.add_argument('--model', type=str, default=None, 
                help=f'Ollama model to use (default: from env or {OLLAMA_MODEL})')
    args = parser.parse_args()

    model = args.model or OLLAMA_MODEL

    # Construct input and output paths
    input_csv = f"output/{args.reviews}"
    
    # Generate output filename by adding '_aspects' before the extension
    filename_parts = args.reviews.rsplit('.', 1)
    if len(filename_parts) == 2:
        output_filename = f"{filename_parts[0]}_aspects.{filename_parts[1]}"
    else:
        output_filename = f"{args.reviews}_aspects"
    output_csv = f"output/{output_filename}"
    
    # Create and run analyzer
    try:
        analyzer = DynamicAspectGrouper(input_csv, model, OLLAMA_HOST)
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return

    analyzer.run(output_csv)

if __name__ == "__main__":
    main()