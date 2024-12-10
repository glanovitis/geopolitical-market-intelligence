import yfinance as yf
import pandas as pd
import requests
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from config import NEWS_API_KEY, MARKET_SYMBOLS, NEWS_SOURCES
from tenacity import retry, stop_after_attempt, wait_exponential


class MarketDataCollector:
    def __init__(self, financial_symbols=MARKET_SYMBOLS, news_sources=NEWS_SOURCES, days_of_history=30):
        self.financial_symbols = financial_symbols
        self.news_sources = news_sources
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=days_of_history)).strftime('%Y-%m-%d')

        if not NEWS_API_KEY:
            raise ValueError("NEWS_API_KEY not found in environment variables")

        # Initialize News API with key from config
        self.newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    def collect_market_data(self):
        """
        Collect historical financial data for specified symbols
        """
        market_data = {}
        
        for symbol in self.financial_symbols:
            try:
                # Download historical stock data
                stock = yf.Ticker(symbol)
                historical_data = stock.history(
                    start=self.start_date, 
                    end=self.end_date
                )
                
                # Additional financial metrics
                market_data[symbol] = {
                    'price_history': historical_data,
                    'info': stock.info,
                    'financials': stock.financials,
                    'earnings': stock.income_stmt.loc['Net Income']
                }
            
            except Exception as e:
                print(f"Error collecting data for {symbol}: {e}")
        
        return market_data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def collect_news_data(self):
        """Collect news articles with retry mechanism"""
        """Collect news articles related to financial markets"""
        news_data = []
        
        # Iterate through predefined news sources
        for source in self.news_sources:
            try:
                # Fetch articles
                articles = self.newsapi.get_everything(
                    sources=source,
                    from_param=self.start_date,
                    to=self.end_date,
                    language='en',
                    sort_by='relevancy'
                )
                
                # Process and store articles
                for article in articles['articles']:
                    news_data.append({
                        'source': article['source']['name'],
                        'title': article['title'],
                        'description': article['description'],
                        'published_at': article['publishedAt'],
                        'url': article['url']
                    })
            
            except Exception as e:
                print(f"Error collecting news from {source}: {e}")
        
        return news_data

    def prepare_combined_dataset(self):
        """
        Combine market and news data
        """
        market_data = self.collect_market_data()
        news_data = self.collect_news_data()
        
        # Create a comprehensive dataset
        combined_dataset = {
            'market_data': market_data,
            'news_data': news_data
        }
        
        return combined_dataset

    def save_to_csv(self, dataset):
        """
        Save collected data to CSV files
        """
        # Market data
        for symbol, data in dataset['market_data'].items():
            data['price_history'].to_csv(f'{symbol}_price_history.csv')
        
        # News data
        news_df = pd.DataFrame(dataset['news_data'])
        news_df.to_csv('news_data.csv', index=False)

# Example Usage
def main():
    # Define parameters
    financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    news_sources = [
        'bloomberg',
        'business-insider',
        'reuters',
        'bbc-news',
        'cnn'
    ]
    # Get current date
    end_date = datetime.now().strftime('%Y-%m-%d')
    # Get date 30 days ago
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    # Create data collector
    collector = MarketDataCollector(
        financial_symbols, 
        news_sources, 
        start_date, 
        end_date
    )

    # Collect and save data
    dataset = collector.prepare_combined_dataset()
    collector.save_to_csv(dataset)

if __name__ == "__main__":
    main()
