# src/portfolio/data_handler.py
"""
Data fetching and validation for portfolio optimization
Handles WRDS connections, data quality checks, and ticker validation
"""

import pandas as pd
import numpy as np
import wrds
import time
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime

class DataHandler:
    """
    Handles data fetching, validation, and quality control for portfolio optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._db_connection = None
        
    def connect_to_wrds(self, max_retries: int = 3) -> Optional[wrds.Connection]:
        """
        Establish connection to WRDS database with enhanced error handling
        
        Parameters:
        -----------
        max_retries : int
            Maximum number of connection attempts (default: 3)
            
        Returns:
        --------
        wrds.Connection or None
            WRDS connection object if successful, None otherwise
        """
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting WRDS connection (attempt {attempt + 1}/{max_retries})...")
                db = wrds.Connection()
                self.logger.info("WRDS connection successful")
                self._db_connection = db
                return db
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Connection failed after {max_retries} attempts: {e}")
                    self.logger.error("Please check your WRDS credentials and internet connection")
                    return None
                else:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    self.logger.info("Retrying...")
                    time.sleep(2)  # Wait before retry
        return None
        
    def fetch_stock_returns(self, 
                          tickers: List[str], 
                          start_date: str, 
                          end_date: str,
                          db_connection: Optional[wrds.Connection] = None) -> Optional[pd.DataFrame]:
        """
        Query WRDS CRSP database for monthly stock return data
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock tickers
        start_date : str
            Start date string 'YYYY-MM-DD'
        end_date : str
            End date string 'YYYY-MM-DD'
        db_connection : wrds.Connection, optional
            WRDS connection object. If None, will create new connection.
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame with dates as index and tickers as columns
        """
        
        # Use provided connection or create new one
        if db_connection is None:
            db = self.connect_to_wrds()
            if db is None:
                return None
        else:
            db = db_connection
            
        # Convert tickers to string for SQL query
        ticker_str = "', '".join(tickers)
        
        # SQL query for CRSP monthly stock file with proper date handling
        query = f"""
        SELECT date, ticker, ret
        FROM crsp.msf a
        LEFT JOIN crsp.msenames b
        ON a.permno = b.permno
        WHERE DATE_TRUNC('month', b.namedt) <= DATE_TRUNC('month', a.date) 
        AND DATE_TRUNC('month', a.date) <= DATE_TRUNC('month', b.nameendt)
        AND a.date BETWEEN '{start_date}' AND '{end_date}'
        AND ticker IN ('{ticker_str}')
        AND ret IS NOT NULL
        ORDER BY date, ticker
        """
        
        try:
            self.logger.info(f"Executing CRSP query for {len(tickers)} tickers...")
            
            # Execute query
            data = db.raw_sql(query)
            
            if data.empty:
                self.logger.warning("No data returned from CRSP query")
                return None
                
            # Convert to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Remove duplicates by keeping the first occurrence for each date-ticker pair
            data = data.drop_duplicates(subset=['date', 'ticker'], keep='first')
            
            # Pivot to get tickers as columns
            returns_df = data.pivot(index='date', columns='ticker', values='ret')
            
            # Sort by date
            returns_df = returns_df.sort_index()
            
            self.logger.info(f"Successfully fetched data: {len(returns_df)} periods, {len(returns_df.columns)} tickers")
            
            # Close connection if we created it
            if db_connection is None:
                db.close()
                
            return returns_df
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            
            # Close connection if we created it
            if db_connection is None and 'db' in locals():
                db.close()
                
            return None
            
    def validate_ticker_coverage(self, 
                                returns_df: pd.DataFrame, 
                                ticker_list: List[str], 
                                min_observations: int, 
                                max_missing_pct: float) -> Tuple[List[str], List[str]]:
        """
        Validate if tickers have sufficient data coverage within the time period
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame with return data
        ticker_list : List[str]
            List of tickers to validate
        min_observations : int
            Minimum required observations per ticker
        max_missing_pct : float
            Maximum allowed missing data percentage (0.0 to 1.0)
            
        Returns:
        --------
        Tuple[List[str], List[str]]
            (valid_tickers, insufficient_tickers)
        """
        
        valid_tickers = []
        insufficient_tickers = []
        
        self.logger.info(f"Validating data coverage for {len(ticker_list)} tickers...")
        
        # Loop through each ticker to find the coverage
        for ticker in ticker_list:
            if ticker not in returns_df.columns:
                insufficient_tickers.append(ticker)
                self.logger.warning(f"Ticker {ticker} not found in data")
                continue
                
            # Calculate coverage stats
            series = returns_df[ticker]
            total_periods = len(returns_df)
            available_periods = series.notna().sum()
            missing_pct = ((total_periods - available_periods) / total_periods)
            
            # Check coverage requirements
            passes_min_obs = available_periods >= min_observations
            passes_missing_pct = missing_pct <= max_missing_pct
            
            if passes_min_obs and passes_missing_pct:
                valid_tickers.append(ticker)
                self.logger.debug(f"Ticker {ticker}: {available_periods}/{total_periods} obs ({missing_pct:.1%} missing) - VALID")
            else:
                insufficient_tickers.append(ticker)
                self.logger.warning(f"Ticker {ticker}: {available_periods}/{total_periods} obs ({missing_pct:.1%} missing) - INSUFFICIENT")
                
        # Log summary
        if not insufficient_tickers:
            self.logger.info("Data validation: All tickers have sufficient data")
        else:
            self.logger.warning(f"Data validation: {len(insufficient_tickers)} ticker(s) have insufficient data")
            
        return valid_tickers, insufficient_tickers
        
    def get_default_tickers(self) -> List[str]:
        """
        Get default ticker list for fallback scenarios
        
        Returns:
        --------
        List[str]
            List of 20 default blue-chip tickers
        """
        
        return [
            'AAPL', 'MSFT', 'GOOG', 'IBM', 'NVDA',   # Technology
            'JPM', 'BAC', 'MS', 'GS',                # Financial
            'JNJ', 'PFE', 'MRK',                     # Healthcare
            'KO', 'PG', 'WMT', 'HD',                 # Consumer
            'GE', 'CAT', 'MMM',                      # Industrial
            'XOM'                                    # Energy
        ]
        
    def suggest_replacement_tickers(self) -> List[str]:
        """
        Suggest alternative tickers for replacement when data is insufficient
        
        Returns:
        --------
        List[str]
            Pool of replacement ticker candidates
        """
        
        replacement_pool = [
            # Additional tech
            'ORCL', 'CRM', 'INTC', 'AMD', 'ADBE',
            # Additional financial
            'C', 'AXP', 'BLK', 'WFC',
            # Additional consumer
            'AMZN', 'NFLX', 'DIS', 'NKE', 'SBUX',
            # Additional healthcare
            'UNH', 'ABBV', 'TMO', 'ABT',
            # Additional industrial
            'HON', 'UNP', 'LMT', 'BA',
            # Utilities
            'NEE', 'DUK', 'SO',
            # Materials
            'LIN', 'APD'
        ]
        return replacement_pool
        
    def interactive_ticker_replacement(self, 
                                     insufficient_tickers: List[str], 
                                     replacement_pool: List[str], 
                                     db_connection: wrds.Connection, 
                                     returns_df: pd.DataFrame, 
                                     config: Dict) -> List[str]:
        """
        Interactive loop for replacing insufficient tickers
        
        Parameters:
        -----------
        insufficient_tickers : List[str]
            Tickers that failed validation
        replacement_pool : List[str]
            Pool of replacement candidates
        db_connection : wrds.Connection
            WRDS database connection
        returns_df : pd.DataFrame
            Current returns data
        config : Dict
            Configuration parameters
            
        Returns:
        --------
        List[str]
            Updated list of valid tickers
        """
        
        if not insufficient_tickers:
            return config['tickers']
            
        self.logger.info(f"\n{len(insufficient_tickers)} tickers failed validation: {insufficient_tickers}")
        print(f"\n{len(insufficient_tickers)} tickers failed validation: {insufficient_tickers}")
        print("Options:")
        print("1. Remove insufficient tickers and continue")
        print("2. Replace with suggested alternatives")
        print("3. Manually specify replacement tickers")
        
        choice = input("Choose option (1/2/3): ").strip()
        
        updated_tickers = [ticker for ticker in config['tickers'] if ticker not in insufficient_tickers]
        
        if choice == '1':
            print(f"Continuing with {len(updated_tickers)} valid tickers")
            final_tickers = updated_tickers
            
        elif choice == '2':
            print("\nSuggested replacement tickers:")
            available_replacements = [t for t in replacement_pool if t not in config['tickers']]
            
            for i, ticker in enumerate(available_replacements[:10], 1):
                print(f"{i:2d}. {ticker}")
                
            selections = input("\nSelect replacements by number (e.g., 1,3,5): ").strip()
            
            if selections:
                try:
                    indices = [int(x.strip()) - 1 for x in selections.split(',')]
                    selected_tickers = [available_replacements[i] for i in indices 
                                      if 0 <= i < len(available_replacements)]
                    updated_tickers.extend(selected_tickers)
                    print(f"Added replacements: {selected_tickers}")
                except (ValueError, IndexError):
                    print("Invalid selection, continuing without replacements")
                    
            final_tickers = updated_tickers
            
        elif choice == '3':
            manual_tickers = input("Enter replacement tickers (space-separated): ").strip().upper().split()
            if manual_tickers:
                updated_tickers.extend(manual_tickers)
                print(f"Added manual replacements: {manual_tickers}")
                
            final_tickers = updated_tickers
        else:
            final_tickers = updated_tickers
            
        # Validate new tickers if any were added
        new_tickers = [t for t in final_tickers if t not in config['tickers'] or t in insufficient_tickers]
        
        if new_tickers:
            print(f"\nValidating new tickers: {new_tickers}")
            
            # Fetch data for new tickers
            start_date = f"{config['start_year']}-01-01"
            end_date = f"{config['end_year']}-12-31"
            
            new_data = self.fetch_stock_returns(new_tickers, start_date, end_date, db_connection)
            
            if new_data is not None:
                # Merge with existing data
                combined_data = returns_df.copy()
                for ticker in new_data.columns:
                    combined_data[ticker] = new_data[ticker]
                    
                # Validate new tickers
                valid_new, invalid_new = self.validate_ticker_coverage(
                    combined_data, new_tickers,
                    config['coverage_params']['min_observations'],
                    config['coverage_params']['max_missing_pct']
                )
                
                # Keep only valid new tickers
                final_tickers = [t for t in final_tickers if t not in new_tickers or t in valid_new]
            else:
                # If new data fetch failed, remove new tickers
                final_tickers = [t for t in final_tickers if t not in new_tickers]
                
        # Ensure minimum ticker count
        if len(final_tickers) < 1:
            self.logger.warning("No valid tickers remaining. Adding default tickers.")
            default_tickers = self.get_default_tickers()
            final_tickers.extend([t for t in default_tickers if t not in final_tickers])
            
        return final_tickers
        
    def load_static_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load returns data from static file (CSV, parquet, etc.)
        Alternative to WRDS for development/demo purposes

        Parameters:
        -----------
        file_path : str
            Path to data file

        Returns:
        --------
        pd.DataFrame or None
            Returns data with dates as index and tickers as columns
        """

        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            self.logger.info(f"Loaded static data: {len(data)} periods, {len(data.columns)} tickers")
            return data

        except Exception as e:
            self.logger.error(f"Error loading static data: {e}")
            return None