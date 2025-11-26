# src/utils/validation.py
"""
Input validation utilities for portfolio optimization
Handles parameter validation, data quality checks, and constraint verification
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime

class InputValidator:
    """
    Validates user inputs and configuration parameters for portfolio optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_tickers(self, tickers: Union[List[str], str]) -> List[str]:
        """
        Validate and clean ticker list
        
        Parameters:
        -----------
        tickers : Union[List[str], str]
            List of tickers or space-separated string
            
        Returns:
        --------
        List[str]
            Cleaned and validated ticker list
        """
        
        # Convert string to list if necessary
        if isinstance(tickers, str):
            tickers = tickers.upper().split()
        elif isinstance(tickers, list):
            tickers = [t.upper().strip() for t in tickers if isinstance(t, str)]
        else:
            raise ValueError("Tickers must be a list of strings or space-separated string")
            
        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []
        duplicates = []
        
        for ticker in tickers:
            if ticker in seen:
                duplicates.append(ticker)
            else:
                unique_tickers.append(ticker)
                seen.add(ticker)
                
        if duplicates:
            self.logger.warning(f"Removed duplicate tickers: {duplicates}")
            
        # Basic ticker format validation
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in unique_tickers:
            if self._is_valid_ticker_format(ticker):
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
                
        if invalid_tickers:
            self.logger.warning(f"Invalid ticker formats removed: {invalid_tickers}")
            
        if len(valid_tickers) == 0:
            raise ValueError("No valid tickers provided")
            
        return valid_tickers
        
    def validate_date_range(self, start_year: int, end_year: int) -> Tuple[int, int]:
        """
        Validate date range for analysis
        
        Parameters:
        -----------
        start_year : int
            Start year for analysis
        end_year : int
            End year for analysis
            
        Returns:
        --------
        Tuple[int, int]
            Validated (start_year, end_year)
        """
        
        current_year = datetime.now().year
        
        # Basic range checks
        if start_year < 1990:
            raise ValueError("Start year must be 1990 or later (data quality concerns)")
        if start_year > current_year:
            raise ValueError(f"Start year cannot be in the future (current: {current_year})")
        if end_year < start_year:
            raise ValueError("End year must be greater than start year")
        if end_year > current_year:
            self.logger.warning(f"End year {end_year} is in the future, using {current_year}")
            end_year = current_year
            
        # Check minimum period length
        period_years = end_year - start_year
        if period_years < 2:
            raise ValueError(f"Analysis period too short ({period_years} years). Minimum 2 years required.")
        if period_years < 3:
            self.logger.warning(f"Short analysis period ({period_years} years). Consider at least 3 years for robust results.")
            
        return start_year, end_year
        
    def validate_estimation_window(self, estimation_window: int, analysis_period_years: int) -> int:
        """
        Validate estimation window parameter
        
        Parameters:
        -----------
        estimation_window : int
            Estimation window in months
        analysis_period_years : int
            Total analysis period in years
            
        Returns:
        --------
        int
            Validated estimation window
        """
        
        if estimation_window < 6:
            raise ValueError("Estimation window must be at least 6 months")
        if estimation_window > 240:
            raise ValueError("Estimation window cannot exceed 240 months (20 years)")
            
        # Check against analysis period
        analysis_months = analysis_period_years * 12
        min_required_months = estimation_window + 24  # Need buffer for backtest
        
        if analysis_months < min_required_months:
            self.logger.warning(
                f"Short analysis period ({analysis_months} months) for estimation window ({estimation_window} months). "
                f"Consider extending period or reducing window."
            )
            
        # Provide guidance on typical windows
        if estimation_window < 12:
            self.logger.info("Short estimation window may lead to unstable covariance estimates")
        elif estimation_window > 60:
            self.logger.info("Long estimation window may be slow to adapt to changing market conditions")
            
        return estimation_window
        
    def validate_constraints(self, constraints: Dict) -> Dict:
        """
        Validate portfolio constraints
        
        Parameters:
        -----------
        constraints : Dict
            Portfolio constraint parameters
            
        Returns:
        --------
        Dict
            Validated constraints dictionary
        """
        
        validated = constraints.copy()
        
        # Validate weight bounds
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        
        if min_weight > max_weight:
            raise ValueError(f"min_weight ({min_weight}) cannot exceed max_weight ({max_weight})")
            
        if min_weight < -1.0:
            self.logger.warning("Very negative min_weight may lead to extreme short positions")
        if max_weight > 1.0:
            self.logger.warning("max_weight above 100% allows leveraged positions")
            
        # Validate consistency with long_only flag
        long_only = constraints.get('long_only', False)
        allow_short = constraints.get('allow_short', True)
        
        if long_only and allow_short:
            self.logger.warning("Conflicting constraints: long_only=True but allow_short=True. Setting allow_short=False")
            validated['allow_short'] = False
            
        if long_only and min_weight < 0:
            self.logger.warning("long_only=True but min_weight < 0. Setting min_weight=0")
            validated['min_weight'] = 0.0
            
        # Check feasibility
        if long_only and max_weight < 1.0:
            num_assets_needed = int(np.ceil(1.0 / max_weight))
            self.logger.info(f"With max_weight={max_weight}, need at least {num_assets_needed} assets for feasible portfolio")
            
        return validated
        
    def validate_risk_free_rate(self, risk_free_rate: float) -> float:
        """
        Validate risk-free rate parameter
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        float
            Validated risk-free rate
        """
        
        if risk_free_rate < 0:
            self.logger.warning("Negative risk-free rate - unusual but allowed")
        elif risk_free_rate > 0.20:
            self.logger.warning(f"Very high risk-free rate ({risk_free_rate:.1%}) - please verify")
        elif risk_free_rate > 0.10:
            self.logger.info(f"High risk-free rate ({risk_free_rate:.1%}) for typical developed markets")
            
        return risk_free_rate
        
    def validate_coverage_params(self, coverage_params: Dict, estimation_window: int) -> Dict:
        """
        Validate data coverage parameters
        
        Parameters:
        -----------
        coverage_params : Dict
            Data quality requirements
        estimation_window : int
            Estimation window in months
            
        Returns:
        --------
        Dict
            Validated coverage parameters
        """
        
        validated = coverage_params.copy()
        
        min_observations = coverage_params.get('min_observations', estimation_window)
        max_missing_pct = coverage_params.get('max_missing_pct', 0.10)
        
        # Validate min_observations
        if min_observations < 6:
            self.logger.warning("Very low min_observations may lead to unstable estimates")
        if min_observations > estimation_window:
            self.logger.warning(f"min_observations ({min_observations}) exceeds estimation_window ({estimation_window})")
            
        # Validate max_missing_pct
        if max_missing_pct < 0 or max_missing_pct > 1:
            raise ValueError("max_missing_pct must be between 0 and 1")
        if max_missing_pct > 0.3:
            self.logger.warning(f"High tolerance for missing data ({max_missing_pct:.0%}) may reduce portfolio quality")
            
        validated['min_observations'] = min_observations
        validated['max_missing_pct'] = max_missing_pct
        
        return validated
        
    def validate_optimization_config(self, config: Dict) -> Dict:
        """
        Comprehensive validation of complete optimization configuration
        
        Parameters:
        -----------
        config : Dict
            Complete configuration dictionary
            
        Returns:
        --------
        Dict
            Validated configuration
        """
        
        self.logger.info("Validating optimization configuration...")
        
        validated_config = {}
        
        # Validate tickers
        validated_config['tickers'] = self.validate_tickers(config['tickers'])
        
        # Validate date range
        start_year, end_year = self.validate_date_range(config['start_year'], config['end_year'])
        validated_config['start_year'] = start_year
        validated_config['end_year'] = end_year
        
        # Validate estimation window
        period_years = end_year - start_year
        validated_config['estimation_window'] = self.validate_estimation_window(
            config['estimation_window'], period_years
        )
        
        # Validate constraints
        validated_config['constraints'] = self.validate_constraints(config.get('constraints', {}))
        
        # Validate risk-free rate
        validated_config['risk_free_rate'] = self.validate_risk_free_rate(
            config.get('risk_free_rate', 0.042)
        )
        
        # Validate coverage parameters
        validated_config['coverage_params'] = self.validate_coverage_params(
            config.get('coverage_params', {}), validated_config['estimation_window']
        )
        
        # Copy other parameters
        for key, value in config.items():
            if key not in validated_config:
                validated_config[key] = value
                
        self.logger.info("Configuration validation completed successfully")
        
        return validated_config
        
    def _is_valid_ticker_format(self, ticker: str) -> bool:
        """
        Basic ticker format validation
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to validate
            
        Returns:
        --------
        bool
            True if ticker format appears valid
        """
        
        if not isinstance(ticker, str):
            return False
            
        # Basic checks
        if len(ticker) < 1 or len(ticker) > 6:
            return False
            
        # Should be alphanumeric
        if not ticker.replace('.', '').replace('-', '').isalnum():
            return False
            
        # Should not be all numbers
        if ticker.isdigit():
            return False
            
        return True