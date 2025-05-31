#!/usr/bin/env python3
"""
Steam Review Sentiment Analyzer
Fetches reviews from Steam and performs sentiment analysis using local LLM via Ollama
"""
import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests
from tqdm import tqdm
import ollama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables with fallbacks
STEAM_API_KEY = os.getenv('STEAM_API_KEY', '')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral:instruct')
REVIEWS_PER_REQUEST = int(os.getenv('REVIEWS_PER_REQUEST', '100'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '25'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

class SteamReviewAnalyzer:
    def __init__(self, api_key: str, model: str = OLLAMA_MODEL, ollama_host: str = OLLAMA_HOST):
        if not api_key:
            raise ValueError("Steam API key is required. Please set STEAM_API_KEY in your .env file.")
        
        self.api_key = api_key
        self.model = model
        self.ollama_host = ollama_host
        self.base_url = "https://store.steampowered.com/appreviews"
        
        # Batch processing statistics
        self.batch_stats = {
            'successful_batches': 0,
            'failed_batches': 0,
            'fallback_reviews': 0,
            'total_batches': 0
        }
        
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

    def fetch_reviews(self, app_id: str, num_reviews: Optional[int] = None) -> List[Dict]:
        """Fetch reviews from Steam API with pagination"""
        reviews = []
        cursor = "*"
        fetch_all = num_reviews is None
        target_count = num_reviews or float('inf')
        
        print(f"Fetching {'all' if fetch_all else num_reviews} reviews for app ID: {app_id}")
        
        # Use progress bar only if we have a specific target
        pbar = tqdm(total=num_reviews, desc="Fetching reviews") if num_reviews else None
        
        try:
            while len(reviews) < target_count:
                params = {
                    'json': 1,
                    'key': self.api_key,
                    'filter': 'all',
                    'language': 'all',
                    'num_per_page': REVIEWS_PER_REQUEST,
                    'cursor': cursor,
                    'purchase_type': 'all'
                }
                
                try:
                    response = requests.get(f"{self.base_url}/{app_id}", params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data['success'] != 1:
                        print(f"API request failed: {data}")
                        break
                    
                    batch_reviews = data.get('reviews', [])
                    if not batch_reviews:
                        print("\n✓ Fetched all available reviews")
                        break
                    
                    reviews.extend(batch_reviews)
                    
                    if pbar:
                        pbar.update(len(batch_reviews))
                    else:
                        print(f"\rFetched {len(reviews)} reviews...", end='', flush=True)
                    
                    cursor = data.get('cursor', '*')
                    if cursor == '*' or not cursor:
                        print("\n✓ Reached end of reviews")
                        break
                    
                    time.sleep(float(os.getenv('REQUEST_DELAY', '0.5')))  # Rate limiting from env
                    
                except requests.exceptions.RequestException as e:
                    print(f"\nError fetching reviews: {e}")
                    break
                    
        finally:
            if pbar:
                pbar.close()
            else:
                print()  # New line after progress
        
        # Trim to requested number if specified
        if num_reviews and len(reviews) > num_reviews:
            reviews = reviews[:num_reviews]
            
        return reviews

    def create_improved_batch_analysis_prompt(self, review_batch: List[Dict]) -> str:
        """Create an improved analysis prompt for better batch processing reliability"""
        reviews_text = ""
        max_review_length = int(os.getenv('MAX_REVIEW_LENGTH', '600'))  # Reduced from 800
        
        for i, review in enumerate(review_batch, 1):
            # Clean and truncate review text
            review_text = review['review'].replace('\n', ' ').replace('\r', ' ')
            review_text = re.sub(r'\s+', ' ', review_text).strip()
            reviews_text += f"\nREVIEW_{i}:\n{review_text[:max_review_length]}\n"
        
        prompt = f"""You are analyzing {len(review_batch)} game reviews. For each review, provide analysis in EXACTLY the format specified.

IMPORTANT: 
- Respond with ONLY a valid JSON array
- Include exactly {len(review_batch)} objects in the array
- Follow the exact order of reviews
- Use only "positive", "negative", or "mixed" for sentiment
- Keep aspects concise (max 3-4 words each)

Reviews to analyze:
{reviews_text}

Required JSON format (respond with ONLY this JSON, no other text):
[
{', '.join([f'''
  {{
    "review_id": "REVIEW_{i+1}",
    "translation": null,
    "sentiment": "positive",
    "positive_aspects": ["gameplay", "graphics"],
    "negative_aspects": ["bugs", "price"]
  }}''' for i in range(len(review_batch))])}
]"""
        
        return prompt

    def analyze_review_batch_with_retry(self, review_batch: List[Dict], max_retries: int = 3) -> List[Dict]:
        """Analyze a batch of reviews with retry logic and improved error handling"""
        self.batch_stats['total_batches'] += 1
        
        for attempt in range(max_retries):
            try:
                # Adjust parameters based on attempt
                if attempt == 0:
                    # First attempt: normal parameters
                    max_tokens = int(os.getenv('LLM_MAX_TOKENS_BATCH', '4000'))
                    temperature = float(os.getenv('LLM_TEMPERATURE', '0.2'))  # Lower temperature for consistency
                elif attempt == 1:
                    # Second attempt: more conservative parameters
                    max_tokens = int(os.getenv('LLM_MAX_TOKENS_BATCH', '4000')) + 1000
                    temperature = 0.1  # Even lower temperature
                else:
                    # Final attempt: most conservative
                    max_tokens = int(os.getenv('LLM_MAX_TOKENS_BATCH', '4000')) + 2000
                    temperature = 0.05
                
                if attempt > 0:
                    print(f"\n  Retry attempt {attempt + 1}/{max_retries} for batch...")
                    time.sleep(1)  # Brief delay before retry
                
                prompt = self.create_improved_batch_analysis_prompt(review_batch)
                
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=False,
                    options={
                        'temperature': temperature,
                        'top_p': float(os.getenv('LLM_TOP_P', '0.9')),
                        'max_tokens': max_tokens,
                        'num_ctx': int(os.getenv('LLM_CONTEXT_SIZE', '8192')),
                        'repeat_penalty': 1.1,  # Reduce repetition
                        'top_k': 40  # More focused responses
                    }
                )
                
                response_text = response['response'].strip()
                
                # More robust JSON extraction
                analyses = self._extract_and_validate_batch_json(response_text, len(review_batch))
                
                if analyses and len(analyses) == len(review_batch):
                    # Success! Process the results
                    analyzed_reviews = self._process_batch_analyses(review_batch, analyses)
                    self.batch_stats['successful_batches'] += 1
                    return analyzed_reviews
                else:
                    if attempt < max_retries - 1:
                        print(f"    Batch parsing failed (expected {len(review_batch)}, got {len(analyses) if analyses else 0}), retrying...")
                    continue
                    
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"    JSON decode error on attempt {attempt + 1}: {str(e)[:100]}...")
                continue
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    Error on attempt {attempt + 1}: {str(e)[:100]}...")
                continue
        
        # All retries failed, fall back to individual processing
        print(f"    Batch processing failed after {max_retries} attempts, falling back to individual processing")
        self.batch_stats['failed_batches'] += 1
        self.batch_stats['fallback_reviews'] += len(review_batch)
        return self._fallback_individual_analysis(review_batch)

    def _extract_and_validate_batch_json(self, response_text: str, expected_count: int) -> Optional[List[Dict]]:
        """Extract and validate JSON from batch response with multiple strategies"""
        
        # Strategy 1: Look for complete JSON array
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            try:
                json_str = json_match.group()
                analyses = json.loads(json_str)
                if isinstance(analyses, list) and len(analyses) == expected_count:
                    return analyses
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Clean up common JSON formatting issues
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r'```json\s*|\s*```', '', response_text)
            
            # Find array bounds more flexibly
            start = cleaned.find('[')
            end = cleaned.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = cleaned[start:end]
                
                # Fix common issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before ]
                
                analyses = json.loads(json_str)
                if isinstance(analyses, list) and len(analyses) == expected_count:
                    return analyses
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 3: Try to extract individual objects and reconstruct array
        try:
            objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
            if len(objects) == expected_count:
                analyses = []
                for obj_str in objects:
                    analyses.append(json.loads(obj_str))
                return analyses
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None

    def _process_batch_analyses(self, review_batch: List[Dict], analyses: List[Dict]) -> List[Dict]:
        """Process successful batch analyses into the final format"""
        analyzed_reviews = []
        max_text_length = int(os.getenv('MAX_TEXT_OUTPUT', '1000'))
        
        for review, analysis in zip(review_batch, analyses):
            analyzed_review = {
                'review_id': review['recommendationid'],
                'reviewer_name': review['author']['steamid'],
                'review_text': review['review'][:max_text_length],
                'translation': analysis.get('translation'),
                'sentiment': analysis.get('sentiment', 'unknown').lower(),
                'positive_aspects': ', '.join(analysis.get('positive_aspects', [])),
                'negative_aspects': ', '.join(analysis.get('negative_aspects', []))
            }
            analyzed_reviews.append(analyzed_review)
        
        return analyzed_reviews

    def _fallback_individual_analysis(self, review_batch: List[Dict]) -> List[Dict]:
        """Fallback to individual analysis when batch fails"""
        analyzed_reviews = []
        for review in review_batch:
            analyzed_review = self.analyze_single_review(review)
            analyzed_reviews.append(analyzed_review)
        return analyzed_reviews

    def analyze_single_review(self, review: Dict) -> Dict:
        """Analyze a single review using Ollama"""
        max_review_length = int(os.getenv('MAX_REVIEW_LENGTH', '1000'))
        prompt = f"""Analyze the following game review:
1. Determine if it needs translation to English (if not in English, provide translation)
2. Analyze sentiment (positive, negative, or mixed)
3. Extract key positive aspects mentioned
4. Extract key negative aspects mentioned

Review:
{review['review'][:max_review_length]}

Format your response as JSON:
{{
    "translation": "translated text or null if already in English",
    "sentiment": "positive/negative/mixed",
    "positive_aspects": ["aspect1", "aspect2"],
    "negative_aspects": ["aspect1", "aspect2"]
}}
Provide only the JSON response, no additional text."""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    'temperature': float(os.getenv('LLM_TEMPERATURE', '0.3')),
                    'top_p': float(os.getenv('LLM_TOP_P', '0.9')),
                    'max_tokens': int(os.getenv('LLM_MAX_TOKENS_SINGLE', '2000'))
                }
            )
            
            response_text = response['response']
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
                
                max_text_length = int(os.getenv('MAX_TEXT_OUTPUT', '1000'))
                return {
                    'review_id': review['recommendationid'],
                    'reviewer_name': review['author']['steamid'],
                    'review_text': review['review'][:max_text_length],
                    'translation': analysis.get('translation'),
                    'sentiment': analysis.get('sentiment', 'unknown'),
                    'positive_aspects': ', '.join(analysis.get('positive_aspects', [])),
                    'negative_aspects': ', '.join(analysis.get('negative_aspects', []))
                }
            else:
                return self._create_fallback_analysis(review)
                
        except Exception as e:
            print(f"\nError analyzing single review {review['recommendationid']}: {e}")
            return self._create_fallback_analysis(review)

    def _create_fallback_analysis(self, review: Dict) -> Dict:
        """Create a fallback analysis when LLM fails"""
        max_text_length = int(os.getenv('MAX_TEXT_OUTPUT', '1000'))
        return {
            'review_id': review['recommendationid'],
            'reviewer_name': review['author']['steamid'],
            'review_text': review['review'][:max_text_length],
            'translation': None,
            'sentiment': 'unknown',
            'positive_aspects': '',
            'negative_aspects': ''
        }

    def analyze_reviews(self, reviews: List[Dict], batch_size: int = BATCH_SIZE) -> List[Dict]:
        """Process all reviews in batches or individually"""
        analyzed_reviews = []
        
        if batch_size == 1:
            # Use single processing when batch size is 1
            print(f"\nAnalyzing {len(reviews)} reviews individually...")
            
            for review in tqdm(reviews, desc="Processing reviews"):
                analyzed_review = self.analyze_single_review(review)
                analyzed_reviews.append(analyzed_review)
                
                # Small delay between individual reviews to avoid overwhelming the LLM
                time.sleep(float(os.getenv('ANALYSIS_DELAY', '0.1')))
        else:
            # Use batch processing for batch sizes > 1
            total_batches = (len(reviews) + batch_size - 1) // batch_size
            
            print(f"\nAnalyzing {len(reviews)} reviews in batches of {batch_size}...")
            print(f"Total batches: {total_batches}")
            
            for i in tqdm(range(0, len(reviews), batch_size), desc="Processing batches"):
                batch = reviews[i:i + batch_size]
                batch_results = self.analyze_review_batch_with_retry(batch)
                analyzed_reviews.extend(batch_results)
                
                # Small delay between batches to avoid overwhelming the LLM
                time.sleep(float(os.getenv('ANALYSIS_DELAY', '0.2')))
        
        return analyzed_reviews

    def save_to_csv(self, analyzed_reviews: List[Dict], filename: str):
        """Save analyzed reviews to CSV file"""
        fieldnames = [
            'review_id', 'reviewer_name', 'review_text', 'translation',
            'sentiment', 'positive_aspects', 'negative_aspects'
        ]
        
        output_dir = os.getenv('OUTPUT_DIR', '.')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analyzed_reviews)
        
        print(f"\n✓ Saved {len(analyzed_reviews)} analyzed reviews to {filepath}")

    def generate_summary(self, analyzed_reviews: List[Dict], processing_time: float) -> Dict:
        """Generate summary statistics including processing time"""
        total = len(analyzed_reviews)
        if total == 0:
            return {}
        
        sentiments = {'positive': 0, 'negative': 0, 'mixed': 0, 'unknown': 0}
        for review in analyzed_reviews:
            sentiment = review.get('sentiment', 'unknown').lower()
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
        
        # Calculate time per review
        time_per_review = processing_time / total if total > 0 else 0
        
        summary = {
            'total_reviews': total,
            'positive': sentiments['positive'],
            'negative': sentiments['negative'],
            'mixed': sentiments['mixed'],
            'unknown': sentiments['unknown'],
            'positive_percentage': round(sentiments['positive'] / total * 100, 2),
            'negative_percentage': round(sentiments['negative'] / total * 100, 2),
            'total_processing_time': round(processing_time, 2),
            'processing_time_formatted': self._format_time(processing_time),
            'average_time_per_review': round(time_per_review, 2),
            'batch_stats': self.batch_stats.copy()
        }
        
        return summary
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

def main():
    parser = argparse.ArgumentParser(description='Analyze Steam game reviews using local LLM')
    parser.add_argument('app_id', type=str, help='Steam App ID')
    parser.add_argument('--reviews', type=int, default=None, 
                       help='Number of reviews to analyze (default: all available)')
    parser.add_argument('--model', type=str, default=None, 
                       help=f'Ollama model to use (default: from env or {OLLAMA_MODEL})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Number of reviews per batch (default: from env or {BATCH_SIZE}). Use 1 for individual processing.')
    
    args = parser.parse_args()
    
    # Use environment defaults if not specified via command line
    model = args.model or OLLAMA_MODEL
    batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    
    # Initialize analyzer
    try:
        analyzer = SteamReviewAnalyzer(STEAM_API_KEY, model, OLLAMA_HOST)
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return
    
    # Start total timing
    total_start_time = time.time()
    
    # Fetch reviews
    print("\n" + "="*50)
    print("FETCHING REVIEWS")
    print("="*50)
    fetch_start_time = time.time()
    
    reviews = analyzer.fetch_reviews(args.app_id, args.reviews)
    if not reviews:
        print("No reviews fetched. Check the app ID and API key.")
        return
    
    fetch_time = time.time() - fetch_start_time
    print(f"\n✓ Fetched {len(reviews)} reviews in {analyzer._format_time(fetch_time)}")
    
    # Analyze reviews with specified batch size
    print("\n" + "="*50)
    print("ANALYZING REVIEWS")
    print("="*50)
    analysis_start_time = time.time()
    
    analyzed_reviews = analyzer.analyze_reviews(reviews, batch_size)
    
    analysis_time = time.time() - analysis_start_time
    total_time = time.time() - total_start_time
    
    print(f"\n✓ Analysis completed in {analyzer._format_time(analysis_time)}")
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    review_count = len(analyzed_reviews)
    filename_prefix = os.getenv('OUTPUT_FILENAME_PREFIX', 'review_analysis')
    output_filename = f'{filename_prefix}_{args.app_id}_{review_count}_{timestamp}.csv'
    
    # Save to CSV
    analyzer.save_to_csv(analyzed_reviews, output_filename)
    
    # Print summary
    summary = analyzer.generate_summary(analyzed_reviews, analysis_time)
    if summary:
        print("\n" + "="*50)
        print("📊 ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total reviews analyzed: {summary['total_reviews']}")
        print(f"Positive: {summary['positive']} ({summary['positive_percentage']}%)")
        print(f"Negative: {summary['negative']} ({summary['negative_percentage']}%)")
        print(f"Mixed: {summary['mixed']}")
        print(f"Unknown: {summary['unknown']}")
        
        # Batch processing statistics
        batch_stats = summary['batch_stats']
        if batch_stats['total_batches'] > 0:
            success_rate = (batch_stats['successful_batches'] / batch_stats['total_batches']) * 100
            print(f"\n🔄 BATCH PROCESSING STATS")
            print("-" * 30)
            print(f"Total batches: {batch_stats['total_batches']}")
            print(f"Successful batches: {batch_stats['successful_batches']}")
            print(f"Failed batches: {batch_stats['failed_batches']}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Reviews processed individually: {batch_stats['fallback_reviews']}")
        
        print(f"\n⏱️  TIMING INFORMATION")
        print("-" * 25)
        print(f"Fetch time: {analyzer._format_time(fetch_time)}")
        print(f"Analysis time: {summary['processing_time_formatted']}")
        print(f"Total time: {analyzer._format_time(total_time)}")
        print(f"Average per review: {summary['average_time_per_review']} seconds")

if __name__ == "__main__":
    main()