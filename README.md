# Steam Review Sentiment Analyzer

A comprehensive Python toolkit that leverages local Large Language Models (LLMs) via Ollama to analyze Steam game reviews. Extract sentiment, identify key aspects, and gain deep insights into player opinions with intelligent grouping and statistical analysis.

## 🎯 How It Works

- **Fetches reviews** directly from Steam's API for any game
- **Analyzes sentiment** using local AI models (no cloud dependencies)
- **Extracts aspects** that players love or hate about games
- **Groups similar feedback** intelligently (e.g., "graphics", "visuals", "art style")
- **Provides statistics** and exportable CSV reports

## 🚀 Key Features

- 🎮 **Steam API Integration** - Direct access to review data  
- 🤖 **Local LLM Processing** - Privacy-focused AI analysis via Ollama  
- 📊 **Intelligent Grouping** - Automatically categorizes similar aspects  
- 🔄 **Batch Processing** - Efficient handling of large review datasets  
- 📈 **Statistical Analysis** - Comprehensive sentiment breakdowns  
- 💾 **CSV Export** - Easy data export for further analysis  

## 🎯 For

- **Game Developers** - Understand player feedback at scale
- **Data Analysts** - Extract insights from user-generated content
- **Researchers** - Study gaming communities and sentiment trends
- **Indie Developers** - Analyze competitor reviews and market sentiment
- **Privacy-Conscious Users** - Process data locally without cloud APIs

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Steam API key (get one [here](https://steamcommunity.com/dev/apikey))
- At least 8GB RAM (16GB+ recommended for larger models)

## Installation & Setup

### 1. Install Ollama and instruct model
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral:instruct

# Alternative models you can try:
# ollama pull llama2:7b
# ollama pull codellama:7b
```

### 2. Prepare Environment
```bash
# Copy example .env file
cp .env.example .env

# Replace your Steam API Key in .env file
```

### 3. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Sentiment Analyzer
Fetches and analyzes Steam reviews for sentiment and aspects.

#### Basic usage:
```bash
# Analyze all reviews for a specific Steam game (using App ID)
python sentiment_analyzer.py 730

# Common Steam App IDs:
# 730 = Counter-Strike 2
# 440 = Team Fortress 2
# 570 = Dota 2
# 1086940 = Baldur's Gate 3
```

#### Advanced Usage:
```bash
# Analyze specific number of reviews
python sentiment_analyzer.py 730 --reviews 100

# Use different model
python sentiment_analyzer.py 730 --model llama2:7b

# Use individual processing (slower but more reliable)
python sentiment_analyzer.py 730 --batch-size 1

# Combine multiple options
python sentiment_analyzer.py 730 --reviews 500 --batch-size 10 --model mistral:instruct
```

#### Output:
The script generates a CSV file with columns:
- `review_id` - Unique review identifier
- `reviewer_name` - Steam user ID
- `review_text` - Original review text (truncated)
- `translation` - English translation (if needed)
- `sentiment` - positive/negative/mixed
- `positive_aspects` - Comma-separated positive aspects
- `negative_aspects` - Comma-separated negative aspects

Example output filename: `review_analysis_730_150_20241201_143022.csv`
![Screenshot 2025-05-31 at 15 41 33](https://github.com/user-attachments/assets/c8c0c68e-b7bb-4ca5-91a1-44d832ee5f38)

### 2. Aspect Analyzer
Fetches and analyzes Steam reviews for sentiment and aspects.

#### Basic usage:
```bash
# Analyze aspects from previous output
python aspect_analyzer.py review_analysis_730_150_20241201_143022.csv
```

#### Advanced Usage:
```bash
# Specify custom output file
python aspect_analyzer.py input.csv --output my_grouped_aspects.csv

# Use larger batch size for grouping (faster but may be less accurate)
python aspect_analyzer.py input.csv --batch-size 100

# Skip the group merging step (faster processing)
python aspect_analyzer.py input.csv --skip-merge

# Use different model
python aspect_analyzer.py input.csv --model llama2:7b

# Combine options
python aspect_analyzer.py input.csv --output results.csv --batch-size 30 --model mistral:instruct
```

#### Output:
The script generates a CSV file with columns:
- `aspect_type` - "positive" or "negative"
- `group_name` - Consolidated aspect name
- `total_count` - Total mentions across all similar aspects
- `individual_aspects` - All original aspects grouped together
- `aspect_breakdown` - Individual aspect counts

Example output filename: `review_analysis_730_150_20241201_143022_grouped_aspects_20241201_144530.csv`
![Screenshot 2025-05-31 at 15 41 10](https://github.com/user-attachments/assets/d1e81ffd-9cf0-4156-9011-40b5a43dd515)
