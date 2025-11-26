# src/portfolio/optimizer.py
"""
Main Portfolio Optimizer Class
Handles minimum variance portfolio optimization with rolling window backtesting
"""

import cvxpy as cp
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union

from .data_handler import DataHandler
from .static_data_handler import StaticDataHandler
from .covariance import CovarianceEstimator
from .metrics import PerformanceAnalyzer

warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimizer with Ledoit-Wolf shrinkage estimation
    and rolling window backtesting capabilities.
    """
    
    def __init__(self, 
                 tickers: List[str],
                 start_year: int,
                 end_year: int,
                 estimation_window: int = 36,
                 constraints: Optional[Dict] = None,
                 risk_free_rate: float = 0.042,
                 coverage_params: Optional[Dict] = None):
        """
        Initialize Portfolio Optimizer
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock tickers to optimize
        start_year : int
            Start year for analysis
        end_year : int  
            End year for analysis
        estimation_window : int
            Rolling window size in months (default 36)
        constraints : Dict, optional
            Portfolio constraints (min_weight, max_weight, etc.)
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio (default 4.2%)
        coverage_params : Dict, optional
            Data quality requirements
        """
        
        # Core parameters
        self.tickers = tickers
        self.start_year = start_year
        self.end_year = end_year
        self.estimation_window = estimation_window
        self.risk_free_rate = risk_free_rate
        
        # Set default constraints
        self.constraints = constraints or {
            'min_weight': 0.0,  # No shorts by default
            'max_weight': 1.0,
            'allow_short': False,
            'long_only': True
        }
        
        # Set default coverage parameters
        self.coverage_params = coverage_params or {
            'min_observations': max(24, estimation_window // 2),
            'max_missing_pct': 0.10
        }
        
        # Initialize components
        self.data_handler = DataHandler()
        self.covariance_estimator = CovarianceEstimator()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Results storage
        self.returns_data = None
        self.backtest_results = None
        self.portfolio_weights = None
        self.final_tickers = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def validate_inputs(self) -> bool:
        """Validate input parameters"""
        
        # Check minimum requirements
        if len(self.tickers) < 1:
            raise ValueError("At least 1 ticker required")
            
        if self.end_year <= self.start_year:
            raise ValueError("End year must be greater than start year")
            
        total_years = self.end_year - self.start_year
        if total_years < 2:
            self.logger.warning(f"Short time period ({total_years} years) may limit backtest quality")
            
        if self.estimation_window < 6:
            raise ValueError("Estimation window must be at least 6 months")
            
        return True
        
    def fetch_data(self, db_connection=None) -> pd.DataFrame:
        """
        Fetch stock return data
        
        Parameters:
        -----------
        db_connection : wrds.Connection, optional
            WRDS database connection. If None, will create new connection.
            
        Returns:
        --------
        pd.DataFrame
            Returns data with dates as index and tickers as columns
        """
        
        start_date = f"{self.start_year}-01-01"
        end_date = f"{self.end_year}-12-31"
        
        self.logger.info(f"Fetching data for {len(self.tickers)} assets ({start_date} to {end_date})")
        
        # Use data handler to fetch returns
        self.returns_data = self.data_handler.fetch_stock_returns(
            tickers=self.tickers,
            start_date=start_date,
            end_date=end_date,
            db_connection=db_connection
        )
        
        if self.returns_data is None or self.returns_data.empty:
            raise ValueError("Failed to fetch return data")
            
        return self.returns_data
        
    def validate_tickers(self) -> List[str]:
        """
        Validate ticker data coverage and suggest replacements if needed
        
        Returns:
        --------
        List[str]
            List of valid tickers after validation
        """
        
        if self.returns_data is None:
            raise ValueError("Must fetch data before validating tickers")
            
        # Use data handler for validation
        valid_tickers, insufficient_tickers = self.data_handler.validate_ticker_coverage(
            self.returns_data,
            self.tickers,
            self.coverage_params['min_observations'],
            self.coverage_params['max_missing_pct']
        )
        
        if insufficient_tickers:
            self.logger.warning(f"{len(insufficient_tickers)} tickers failed validation: {insufficient_tickers}")
            
            # For now, just use valid tickers
            # In interactive mode, could prompt for replacements
            self.final_tickers = valid_tickers
        else:
            self.final_tickers = valid_tickers
            
        if len(self.final_tickers) == 0:
            raise ValueError("No valid tickers after validation")
            
        self.logger.info(f"Using {len(self.final_tickers)} valid tickers")
        return self.final_tickers
        
    def generate_backtest_dates(self, start_year: Optional[int] = None) -> List[Tuple]:
        """
        Generate monthly rebalancing dates for rolling window backtest
        
        Parameters:
        -----------
        start_year : int, optional
            Year to start backtest. If None, uses estimation_window + buffer
            
        Returns:
        --------
        List[Tuple]
            List of (estimation_start, estimation_end, oos_date) tuples
        """
        
        if self.returns_data is None:
            raise ValueError("Must fetch data before generating backtest dates")
            
        all_dates = self.returns_data.index.sort_values()
        
        # Determine first out-of-sample date
        if start_year is not None:
            first_oos_date = pd.to_datetime(f"{start_year}-01-01")
            first_oos_date = all_dates[all_dates >= first_oos_date][0]
        else:
            buffer_months = 12
            first_oos_idx = self.estimation_window + buffer_months
            if first_oos_idx >= len(all_dates):
                raise ValueError("Insufficient data for backtest period")
            first_oos_date = all_dates[first_oos_idx]
            
        # Generate monthly rebalancing dates
        backtest_dates = []
        current_oos_date = first_oos_date
        
        while current_oos_date <= all_dates[-1]:
            try:
                estimation_end_idx = all_dates.get_loc(current_oos_date) - 1
                estimation_end = all_dates[estimation_end_idx]
                
                estimation_start_idx = max(0, estimation_end_idx - self.estimation_window + 1)
                estimation_start = all_dates[estimation_start_idx]
                
                actual_window = estimation_end_idx - estimation_start_idx + 1
                if actual_window >= self.estimation_window * 0.9:
                    backtest_dates.append((estimation_start, estimation_end, current_oos_date))
                    
                # Move to next month
                next_month = current_oos_date + relativedelta(months=1)
                future_dates = all_dates[all_dates > current_oos_date]
                if len(future_dates) == 0:
                    break
                current_oos_date = future_dates[0]
                
            except (KeyError, IndexError):
                break
                
        return backtest_dates
        
    def optimize_portfolio(self, 
                          estimation_start: pd.Timestamp,
                          estimation_end: pd.Timestamp,
                          method: str = 'both') -> Dict:
        """
        Optimize portfolio for a single period
        
        Parameters:
        -----------
        estimation_start : pd.Timestamp
            Start date for estimation window
        estimation_end : pd.Timestamp  
            End date for estimation window
        method : str
            Optimization method ('sample', 'ledoit_wolf', or 'both')
            
        Returns:
        --------
        Dict
            Results from optimization with weights, metrics, etc.
        """
        
        if self.final_tickers is None:
            raise ValueError("Must validate tickers before optimization")
            
        filtered_returns = self.returns_data[self.final_tickers].copy()
        
        # Use covariance estimator for optimization
        return self.covariance_estimator.compare_methods(
            returns_df=filtered_returns,
            estimation_start=estimation_start,
            estimation_end=estimation_end,
            constraints=self.constraints,
            method=method
        )
        
    def run_backtest(self, save_progress: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute complete rolling window backtest
        
        Parameters:
        -----------
        save_progress : bool
            Whether to show progress bar
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (backtest_results, portfolio_weights)
        """
        
        # Validate everything is ready
        if self.returns_data is None:
            raise ValueError("Must fetch data before running backtest")
        if self.final_tickers is None:
            raise ValueError("Must validate tickers before running backtest")
            
        # Generate backtest schedule
        backtest_dates = self.generate_backtest_dates()
        
        if not backtest_dates:
            raise ValueError("Could not generate backtest dates")
            
        self.logger.info(f"Running backtest: {len(backtest_dates)} periods")
        
        results_list = []
        weights_list = []
        
        # Progress bar setup
        if save_progress:
            iterator = tqdm(backtest_dates, desc="Processing periods")
        else:
            iterator = backtest_dates
            
        for est_start, est_end, oos_date in iterator:
            try:
                # Get out-of-sample returns
                filtered_returns = self.returns_data[self.final_tickers].copy()
                
                if oos_date not in filtered_returns.index:
                    continue
                    
                oos_returns = filtered_returns.loc[oos_date]
                
                # Skip if too many missing returns
                if oos_returns.isna().sum() > len(self.final_tickers) * 0.2:
                    continue
                    
                # Optimize portfolio
                optimization_results = self.optimize_portfolio(
                    est_start, est_end, method='both'
                )
                
                # Process results for both methods
                period_results = {
                    'date': oos_date,
                    'estimation_start': est_start,
                    'estimation_end': est_end,
                    'estimation_periods': len(filtered_returns.loc[est_start:est_end])
                }
                
                weight_record = {'date': oos_date}
                
                # Process sample and Ledoit-Wolf results
                for method in ['sample', 'ledoit_wolf']:
                    if method in optimization_results and 'error' not in optimization_results[method]:
                        result = optimization_results[method]
                        weights = result['weights']
                        
                        # Calculate out-of-sample return
                        oos_vec = oos_returns.reindex(result['tickers']).fillna(0).values
                        oos_return = float(np.dot(weights, oos_vec))
                        
                        method_prefix = 'sample' if method == 'sample' else 'lw'
                        
                        period_results.update({
                            f'{method_prefix}_return': oos_return,
                            f'{method_prefix}_volatility': result['metrics']['portfolio_volatility_annual'],
                            f'{method_prefix}_num_positions': result['metrics']['num_positions'],
                            f'{method_prefix}_max_position': result['metrics']['max_position'],
                            f'{method_prefix}_long_exposure': result['metrics']['long_exposure'],
                            f'{method_prefix}_short_exposure': result['metrics']['short_exposure'],
                            f'{method_prefix}_gross_exposure': result['metrics']['gross_exposure']
                        })
                        
                        if method == 'ledoit_wolf':
                            period_results[f'{method_prefix}_shrinkage'] = result['shrinkage']
                            
                        # Store weights
                        for ticker, weight in zip(result['tickers'], weights):
                            weight_record[f'{ticker}_{method_prefix}'] = weight
                            
                        # Fill NaN for missing tickers
                        for ticker in self.final_tickers:
                            if f'{ticker}_{method_prefix}' not in weight_record:
                                weight_record[f'{ticker}_{method_prefix}'] = np.nan
                    else:
                        # Fill with NaN if optimization failed
                        method_prefix = 'sample' if method == 'sample' else 'lw'
                        period_results.update({
                            f'{method_prefix}_return': np.nan,
                            f'{method_prefix}_volatility': np.nan,
                            f'{method_prefix}_num_positions': np.nan,
                            f'{method_prefix}_max_position': np.nan,
                            f'{method_prefix}_long_exposure': np.nan,
                            f'{method_prefix}_short_exposure': np.nan,
                            f'{method_prefix}_gross_exposure': np.nan
                        })
                        
                        if method == 'ledoit_wolf':
                            period_results[f'{method_prefix}_shrinkage'] = np.nan
                            
                        for ticker in self.final_tickers:
                            weight_record[f'{ticker}_{method_prefix}'] = np.nan
                            
                results_list.append(period_results)
                weights_list.append(weight_record)
                
            except Exception as e:
                self.logger.warning(f"Error processing period {oos_date}: {e}")
                continue
                
        # Convert to DataFrames
        if not results_list:
            raise ValueError("No valid backtest periods found")
            
        self.backtest_results = pd.DataFrame(results_list)
        self.backtest_results.set_index('date', inplace=True)
        self.backtest_results.sort_index(inplace=True)
        
        self.portfolio_weights = pd.DataFrame(weights_list)
        if len(self.portfolio_weights) > 0:
            self.portfolio_weights.set_index('date', inplace=True)
            self.portfolio_weights.sort_index(inplace=True)
            
        return self.backtest_results, self.portfolio_weights
        
    def analyze_performance(self) -> Dict:
        """
        Analyze backtest performance using PerformanceAnalyzer
        
        Returns:
        --------
        Dict
            Comprehensive performance metrics
        """
        
        if self.backtest_results is None:
            raise ValueError("Must run backtest before analyzing performance")
            
        return self.performance_analyzer.calculate_comprehensive_metrics(
            self.backtest_results,
            self.portfolio_weights,
            self.risk_free_rate
        )
        
    def run_complete_analysis(self, db_connection=None) -> Dict:
        """
        Execute complete portfolio optimization workflow
        
        Parameters:
        -----------
        db_connection : wrds.Connection, optional
            WRDS database connection
            
        Returns:
        --------
        Dict
            Complete results including backtest, performance metrics, etc.
        """
        
        try:
            # 1. Validate inputs
            self.validate_inputs()
            
            # 2. Fetch data
            self.fetch_data(db_connection)
            
            # 3. Validate tickers
            self.validate_tickers()
            
            # 4. Run backtest
            self.run_backtest()
            
            # 5. Analyze performance
            performance_results = self.analyze_performance()
            
            # 6. Compile results
            results = {
                'success': True,
                'config': {
                    'tickers': self.tickers,
                    'final_tickers': self.final_tickers,
                    'start_year': self.start_year,
                    'end_year': self.end_year,
                    'estimation_window': self.estimation_window,
                    'constraints': self.constraints,
                    'risk_free_rate': self.risk_free_rate,
                    'coverage_params': self.coverage_params
                },
                'backtest_results': self.backtest_results,
                'portfolio_weights': self.portfolio_weights,
                'performance_metrics': performance_results['performance_metrics'],
                'turnover_metrics': performance_results.get('turnover_metrics'),
                'stability_metrics': performance_results.get('stability_metrics'),
                'errors': []
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'config': None,
                'backtest_results': None,
                'portfolio_weights': None
            }
            
    def __repr__(self):
        return f"PortfolioOptimizer(tickers={len(self.tickers)}, period={self.start_year}-{self.end_year}, window={self.estimation_window}m)"