# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Add other configuration variables as needed
MARKET_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
NEWS_SOURCES = ['bloomberg', 'business-insider', 'reuters', 'bbc-news', 'cnn']