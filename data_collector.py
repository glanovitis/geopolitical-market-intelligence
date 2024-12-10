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
        self.chunk_size_months = 6  # Process 6 months at a time

    def _get_date_chunks(self, start_date: datetime, end_date: datetime):
        """Break the date range into smaller chunks"""
        chunks = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(
                current_start + relativedelta(months=self.chunk_size_months),
                end_date
            )
            chunks.append((current_start, current_end))
            current_start = current_end
        return chunks

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def collect_historical_news(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        all_articles = []
        date_chunks = self._get_date_chunks(start_date, end_date)

        for chunk_start, chunk_end in date_chunks:
            logging.info(f"Collecting news for period: {chunk_start.date()} to {chunk_end.date()}")
            chunk_articles = self._collect_chunk(chunk_start, chunk_end)
            all_articles.extend(chunk_articles)

            # Save chunk immediately to avoid memory issues
            self._save_chunk(chunk_articles, chunk_start.strftime('%Y%m'))

        return all_articles

    def _collect_chunk(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Collect news for a specific time chunk"""
        articles = []
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

                articles.extend(data['results'])
                logging.info(f"Collected page {current_page}, articles in chunk: {len(articles)}")

                if current_page >= data['pages']:
                    break

                current_page += 1
                time.sleep(self.request_delay)  # Rate limiting

            except Exception as e:
                logging.error(f"Error collecting page {current_page}: {e}")
                if current_page == 1:
                    raise
                break

        return articles

    def _save_chunk(self, articles: List[Dict], chunk_id: str):
        """Save a chunk of articles to temporary storage"""
        if not articles:
            return

        temp_dir = 'data/temp_chunks'
        os.makedirs(temp_dir, exist_ok=True)

        df = pd.DataFrame(articles)
        df.to_csv(f'{temp_dir}/chunk_{chunk_id}.csv', index=False)


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

    def collect_market_data(self):
        """Collect market data for specified symbols"""
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
                logging.info(f"Successfully collected data for {symbol}")
            except Exception as e:
                failed_symbols.append((symbol, str(e)))
                logging.error(f"Failed to collect data for {symbol}: {e}")

        if len(failed_symbols) == len(self.financial_symbols):
            raise RuntimeError("Failed to collect data for all symbols")

        return market_data, failed_symbols

    def collect_news_data(self):
        """Collect news articles from The Guardian"""
        # Initialize empty list for all news
        all_news = []

        try:
            # Call GuardianNewsCollector's collect_historical_news method
            articles = self.news_collector.collect_historical_news(
                start_date=self.start_date,
                end_date=self.end_date
            )
            all_news.extend(articles)
            logging.info(f"Collected {len(articles)} articles from The Guardian")

        except Exception as e:
            logging.error(f"Error collecting news data: {e}")

        return all_news

    def save_to_csv(self, dataset):
        """Save collected data with improved organization and memory handling"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = f'data/collection_{timestamp}'
        os.makedirs(f'{base_dir}/market_data', exist_ok=True)
        os.makedirs(f'{base_dir}/news_data', exist_ok=True)

        # Save market data by year
        for symbol, data in dataset['market_data'].items():
            df = data['price_history']
            df['year'] = df.index.year
            for year, year_data in df.groupby('year'):
                filename = f'{base_dir}/market_data/{symbol}_price_history_{year}.csv'
                year_data.to_csv(filename)
                logging.info(f"Saved market data for {symbol} year {year}")

        # Save news data by year with chunking
        news_df = pd.DataFrame(dataset['news_data'])
        if not news_df.empty:
            news_df['year'] = pd.to_datetime(news_df['webPublicationDate']).dt.year
            for year, year_data in news_df.groupby('year'):
                filename = f'{base_dir}/news_data/news_{year}.csv'
                year_data.to_csv(filename, index=False)
                logging.info(f"Saved {len(year_data)} news articles for year {year}")

        # Save metadata with statistics
        metadata = dataset['metadata']
        metadata.update({
            'statistics': {
                'total_news_articles': len(news_df) if not news_df.empty else 0,
                'news_articles_by_year': news_df['year'].value_counts().to_dict() if not news_df.empty else {},
                'market_data_coverage': {
                    symbol: len(data['price_history'])
                    for symbol, data in dataset['market_data'].items()
                }
            }
        })

        with open(f'{base_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved metadata to {base_dir}/metadata.json")

    def prepare_combined_dataset(self):
        """Combine market and news data"""
        logging.info("Starting to prepare combined dataset...")  # Add debug logging

        logging.info("Collecting market data...")
        market_data, failed_symbols = self.collect_market_data()

        logging.info("Collecting news data...")
        news_data = self.collect_news_data()

        logging.info("Preparing combined dataset structure...")
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

        logging.info("Combined dataset prepared successfully")
        return combined_dataset


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting data collection process...")

    financial_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    years_of_history = 10

    try:
        logging.info("Initializing MarketDataCollector...")
        collector = MarketDataCollector(
            financial_symbols=financial_symbols,
            years_of_history=years_of_history
        )

        logging.info("Preparing combined dataset...")
        dataset = collector.prepare_combined_dataset()

        logging.info("Saving data to CSV...")
        collector.save_to_csv(dataset)

        logging.info("Data collection completed successfully")

    except AttributeError as e:
        logging.error(f"AttributeError: Make sure you're running the latest version of the code. Error: {e}")
        logging.error(f"Available methods: {dir(collector)}")
    except Exception as e:
        logging.error(f"Data collection failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()