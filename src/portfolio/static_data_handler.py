# src/portfolio/static_data_handler.py
"""
Static data handler for portfolio optimization using pre-downloaded CSV data
Handles S&P 500 + NASDAQ + Top ETFs universe without requiring live WRDS connection
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime

class StaticDataHandler:
    """
    Handles pre-downloaded stock and ETF data for portfolio optimization
    Provides same interface as DataHandler but uses static CSV files
    """
    
    def __init__(self, data_file_path: str = 'data/expanded_market_universe_2000_2024.csv'):
        """
        Initialize static data handler
        
        Parameters:
        -----------
        data_file_path : str
            Path to CSV file with return data
        """
        
        self.data_file_path = data_file_path
        self.data = None
        self.available_tickers = None
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> bool:
        """
        Load and prepare the static dataset
        
        Returns:
        --------
        bool
            True if data loaded successfully
        """
        
        try:
            if not os.path.exists(self.data_file_path):
                self.logger.error(f"Data file not found: {self.data_file_path}")
                return False
                
            self.logger.info(f"Loading data from {self.data_file_path}")
            
            # Load CSV data
            raw_data = pd.read_csv(self.data_file_path)
            
            # Expected columns: date, ticker, return, [market_cap, sector]
            required_cols = ['date', 'ticker', 'return']
            if not all(col in raw_data.columns for col in required_cols):
                self.logger.error(f"Missing required columns. Expected: {required_cols}")
                return False
                
            # Convert date column
            raw_data['date'] = pd.to_datetime(raw_data['date'])
            
            # Pivot to wide format (dates as index, tickers as columns)
            self.data = raw_data.pivot(index='date', columns='ticker', values='return')
            self.data.sort_index(inplace=True)
            
            # Get available tickers
            self.available_tickers = list(self.data.columns)
            
            self.logger.info(f"Data loaded successfully: {len(self.data)} periods, {len(self.available_tickers)} tickers")
            self.logger.info(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
            
    def get_available_tickers(self, 
                            min_data_coverage: float = 0.75,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> List[str]:
        """
        Get list of tickers with sufficient data coverage
        
        Parameters:
        -----------
        min_data_coverage : float
            Minimum fraction of non-null observations required
        start_date : str, optional
            Start date for coverage calculation
        end_date : str, optional
            End date for coverage calculation
            
        Returns:
        --------
        List[str]
            List of tickers meeting coverage criteria
        """
        
        if self.data is None:
            if not self.load_data():
                return []
                
        # Filter data by date range if specified
        data_subset = self.data.copy()
        if start_date:
            data_subset = data_subset[data_subset.index >= pd.to_datetime(start_date)]
        if end_date:
            data_subset = data_subset[data_subset.index <= pd.to_datetime(end_date)]
            
        # Calculate coverage for each ticker
        total_periods = len(data_subset)
        coverage = data_subset.notna().sum() / total_periods
        
        # Filter tickers meeting minimum coverage
        valid_tickers = coverage[coverage >= min_data_coverage].index.tolist()
        
        self.logger.info(f"Found {len(valid_tickers)} tickers with >{min_data_coverage:.0%} coverage")
        
        return valid_tickers
        
    def fetch_stock_returns(self, 
                          tickers: List[str], 
                          start_date: str, 
                          end_date: str) -> Optional[pd.DataFrame]:
        """
        Extract returns for specified tickers and date range
        
        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
            
        Returns:
        --------
        pd.DataFrame or None
            Returns data with dates as index and tickers as columns
        """
        
        if self.data is None:
            if not self.load_data():
                return None
                
        try:
            # Convert dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter by date range
            mask = (self.data.index >= start_dt) & (self.data.index <= end_dt)
            filtered_data = self.data.loc[mask]
            
            # Select requested tickers (only those that exist)
            available_tickers = [t for t in tickers if t in filtered_data.columns]
            missing_tickers = [t for t in tickers if t not in filtered_data.columns]
            
            if missing_tickers:
                self.logger.warning(f"Tickers not found in dataset: {missing_tickers}")
                
            if not available_tickers:
                self.logger.error("No requested tickers found in dataset")
                return None
                
            result = filtered_data[available_tickers].copy()
            
            self.logger.info(f"Extracted data: {len(result)} periods, {len(available_tickers)} tickers")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting returns: {e}")
            return None
            
    def validate_ticker_coverage(self, 
                                returns_df: pd.DataFrame, 
                                ticker_list: List[str], 
                                min_observations: int, 
                                max_missing_pct: float) -> Tuple[List[str], List[str]]:
        """
        Validate ticker data coverage (same interface as DataHandler)
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns data
        ticker_list : List[str]
            Tickers to validate
        min_observations : int
            Minimum required observations
        max_missing_pct : float
            Maximum allowed missing data percentage
            
        Returns:
        --------
        Tuple[List[str], List[str]]
            (valid_tickers, insufficient_tickers)
        """
        
        valid_tickers = []
        insufficient_tickers = []
        
        for ticker in ticker_list:
            if ticker not in returns_df.columns:
                insufficient_tickers.append(ticker)
                continue
                
            series = returns_df[ticker]
            total_periods = len(returns_df)
            available_periods = series.notna().sum()
            missing_pct = (total_periods - available_periods) / total_periods
            
            passes_min_obs = available_periods >= min_observations
            passes_missing_pct = missing_pct <= max_missing_pct
            
            if passes_min_obs and passes_missing_pct:
                valid_tickers.append(ticker)
            else:
                insufficient_tickers.append(ticker)
                
        return valid_tickers, insufficient_tickers
        
    def get_universe_by_criteria(self, 
                               sectors: Optional[List[str]] = None,
                               market_cap_min: Optional[float] = None,
                               max_tickers: int = 100,
                               min_data_coverage: float = 0.75) -> List[str]:
        """
        Get investment universe based on criteria
        
        Parameters:
        -----------
        sectors : List[str], optional
            Sector filters (if sector data available)
        market_cap_min : float, optional
            Minimum market cap filter
        max_tickers : int
            Maximum number of tickers to return
        min_data_coverage : float
            Minimum data coverage requirement
            
        Returns:
        --------
        List[str]
            Filtered list of tickers
        """
        
        # Start with all tickers meeting coverage requirements
        valid_tickers = self.get_available_tickers(min_data_coverage)
        
        # For basic implementation, randomly sample or take first N
        # In production, would use sector/market cap data for filtering
        if len(valid_tickers) > max_tickers:
            # Prioritize common tickers (ETFs and large caps tend to have fewer missing values)
            # Simple heuristic: tickers with highest data coverage
            if self.data is not None:
                coverage_scores = self.data[valid_tickers].notna().mean()
                sorted_tickers = coverage_scores.sort_values(ascending=False).index.tolist()
                valid_tickers = sorted_tickers[:max_tickers]
                
        self.logger.info(f"Selected {len(valid_tickers)} tickers based on criteria")
        
        return valid_tickers
        
    def get_etf_list(self) -> List[str]:
        """
        Get list of ETFs in the dataset
        
        Returns:
        --------
        List[str]
            List of ETF tickers (identified by common patterns)
        """
        
        if self.available_tickers is None:
            self.load_data()
            
        # Simple heuristic: common ETF patterns
        etf_patterns = ['SPY', 'VTI', 'QQQ', 'IWM', 'EFA', 'EEM', 'VEA', 'VWO', 'AGG', 'BND']
        common_etfs = [ticker for ticker in self.available_tickers if ticker in etf_patterns]
        
        # Additional pattern matching could be added here
        # (e.g., tickers ending in certain patterns, sector ETFs, etc.)
        
        return common_etfs
        
    def get_stock_list(self) -> List[str]:
        """
        Get list of individual stocks (non-ETFs) in the dataset
        
        Returns:
        --------
        List[str]
            List of stock tickers
        """
        
        if self.available_tickers is None:
            self.load_data()
            
        etfs = self.get_etf_list()
        stocks = [ticker for ticker in self.available_tickers if ticker not in etfs]
        
        return stocks
        
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics about the dataset
        
        Returns:
        --------
        Dict
            Summary information about the dataset
        """
        
        if self.data is None:
            if not self.load_data():
                return {}
                
        etfs = self.get_etf_list()
        stocks = self.get_stock_list()
        
        summary = {
            'total_securities': len(self.available_tickers),
            'stocks': len(stocks),
            'etfs': len(etfs),
            'date_range': {
                'start': self.data.index[0].strftime('%Y-%m-%d'),
                'end': self.data.index[-1].strftime('%Y-%m-%d'),
                'total_periods': len(self.data)
            },
            'data_quality': {
                'avg_coverage': self.data.notna().mean().mean(),
                'min_coverage': self.data.notna().mean().min(),
                'max_coverage': self.data.notna().mean().max()
            }
        }
        
        return summary