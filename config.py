# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')

# config.py
MARKET_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
NEWS_SOURCES = [
    'bloomberg',
    'business-insider',
    'reuters',
    'bbc-news',
    'cnn'
]

# Archive.org configuration
ARCHIVE_RATE_LIMIT = 1  # seconds between requests
ARCHIVE_USER_AGENT = "GeopoliticalMarketIntelligence/1.0"