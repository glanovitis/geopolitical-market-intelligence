import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import os
import json
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from config import GUARDIAN_API_KEY, MARKET_SYMBOLS


class GuardianNewsCollector:
    def __init__(self, api_key: str, keywords: List[str]):
        self.api_key = api_key
        self.keywords = keywords
        self.base_url = "https://content.guardianapis.com/search"
        self.request_delay = 1
        self.max_retries = 3

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def collect_historical_news(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        all_articles = []
        current_page = 1

        params = {
            'api-key': self.api_key,
            'from-date': start_date.strftime('%Y-%m-%d'),
            'to-date': end_date.strftime('%Y-%m-%d'),
            'page-size': 200,
            'show-fields': 'all',
            'order-by': 'oldest',
            'q': ' OR '.join(self.keywords)
        }

        while True:
            try:
                params['page'] = current_page
                response = requests.get(self.base_url, params=params)
                data = response.json()['response']

                if not data['results']:
                    break

                articles = data['results']
                all_articles.extend(articles)

                logging.info(f"Collected page {current_page}, total articles: {len(all_articles)}")

                if current_page >= data['pages']:
                    break

                current_page += 1

            except Exception as e:
                logging.error(f"Error collecting page {current_page}: {e}")
                if current_page == 1:
                    raise
                break

        return all_articles


class MarketDataCollector:
    def __init__(self, financial_symbols=MARKET_SYMBOLS, years_of_history=10):
        self.financial_symbols = financial_symbols
        self.end_date = datetime.now()
        self.start_date = self.end_date - relativedelta(years=years_of_history)

        # Keywords relevant to market and economic news
        self.news_keywords = [
            "global economy",
            "economic growth",
            "financial markets",
            "stock market",
            "international trade",
            "economic policy",
            "central bank",
            "federal reserve",
            "market analysis",
            "global markets"
        ]

        # Initialize Guardian News collector
        if not GUARDIAN_API_KEY:
            raise ValueError("GUARDIAN_API_KEY not found in config")
        self.news_collector = GuardianNewsCollector(GUARDIAN_API_KEY, self.news_keywords)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def collect_market_data(self):
        failed_symbols = []
        market_data = {}

        for symbol in self.financial_symbols:
            try:
                stock = yf.Ticker(symbol)
                historical_data = stock.history(
                    start=self.start_date.strftime('%Y-%m-%d'),
                    end=self.end_date.strftime('%Y-%m-%d')
                )
                if historical_data.empty:
                    raise ValueError(f"No data received for {symbol}")

                market_data[symbol] = {
                    'price_history': historical_data,
                    'info': stock.info
                }
            except Exception as e:
                failed_symbols.append((symbol, str(e)))
                logging.error(f"Failed to collect data for {symbol}: {e}")

        if len(failed_symbols) == len(self.financial_symbols):
            raise RuntimeError("Failed to collect data for all symbols")

        return market_data, failed_symbols

    def collect_news_data(self):
        """Collect news articles from The Guardian"""
        return self.news_collector.collect_historical_news(self.start_date, self.end_date)

    def prepare_combined_dataset(self):
        """Combine market and news data"""
        market_data, failed_symbols = self.collect_market_data()
        news_data = self.collect_news_data()

        combined_dataset = {
            'market_data': market_data,
            'news_data': news_data,
            'metadata': {
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'symbols': self.financial_symbols,
                'failed_symbols': failed_symbols,
                'news_keywords': self.news_keywords
            }
        }

        return combined_dataset

    def save_to_csv(self, dataset):
        """Save collected data to CSV files with better organization"""
        # Create timestamp for the data collection
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = f'data/collection_{timestamp}'
        os.makedirs(f'{base_dir}/market_data', exist_ok=True)
        os.makedirs(f'{base_dir}/news_data', exist_ok=True)

        # Save market data
        for symbol, data in dataset['market_data'].items():
            filename = f'{base_dir}/market_data/{symbol}_price_history.csv'
            data['price_history'].to_csv(filename)
            logging.info(f"Saved market data for {symbol}")

        # Save news data by year
        news_df = pd.DataFrame(dataset['news_data'])
        if not news_df.empty:
            news_df['year'] = pd.to_datetime(news_df['webPublicationDate']).dt.year
            for year, year_data in news_df.groupby('year'):
                filename = f'{base_dir}/news_data/news_{year}.csv'
                year_data.to_csv(filename, index=False)
                logging.info(f"Saved {len(year_data)} news articles for year {year}")

        # Save metadata
        with open(f'{base_dir}/metadata.json', 'w') as f:
            json.dump(dataset['metadata'], f, indent=2)
        logging.info(f"Saved metadata to {base_dir}/metadata.json")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Define parameters
    financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    years_of_history = 10

    # Create data collector
    collector = MarketDataCollector(
        financial_symbols=financial_symbols,
        years_of_history=years_of_history
    )

    try:
        # Collect and save data
        dataset = collector.prepare_combined_dataset()
        collector.save_to_csv(dataset)
        logging.info("Data collection completed successfully")
    except Exception as e:
        logging.error(f"Data collection failed: {e}")


if __name__ == "__main__":
    main()