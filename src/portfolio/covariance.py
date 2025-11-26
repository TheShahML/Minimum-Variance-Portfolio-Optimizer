# src/portfolio/covariance.py
"""
Covariance estimation and portfolio optimization methods
Implements sample covariance and Ledoit-Wolf shrinkage estimation
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import logging
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Tuple, Optional, Union

class CovarianceEstimator:
    """
    Handles covariance matrix estimation and portfolio optimization
    Supports both sample covariance and Ledoit-Wolf shrinkage methods
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_sample_covariance(self, 
                                  returns_df: pd.DataFrame, 
                                  estimation_start: pd.Timestamp, 
                                  estimation_end: pd.Timestamp) -> Tuple[np.ndarray, pd.Series, List[str]]:
        """
        Calculate sample covariance matrix from estimation period stock return data
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame with returns data
        estimation_start : pd.Timestamp
            Start date for estimation window
        estimation_end : pd.Timestamp
            End date for estimation window
            
        Returns:
        --------
        Tuple[np.ndarray, pd.Series, List[str]]
            (sample_cov, mean_returns, valid_tickers)
        """
        
        # Extract estimation period data
        estimation_data = returns_df.loc[estimation_start:estimation_end].copy()
        
        # Remove tickers with insufficient data in this window
        min_obs_threshold = int(len(estimation_data) * 0.75)  # Need at least 75% of periods
        valid_tickers = []
        
        for ticker in estimation_data.columns:
            if estimation_data[ticker].notna().sum() >= min_obs_threshold:
                valid_tickers.append(ticker)
                
        if len(valid_tickers) == 0:
            raise ValueError("No tickers have sufficient data in estimation window")
            
        # Keep only valid tickers and forward-fill missing values
        clean_data = estimation_data[valid_tickers].ffill().dropna()
        
        if clean_data.empty:
            raise ValueError("No valid data after cleaning")
            
        # Calculate sample covariance matrix
        sample_cov = clean_data.cov().values
        mean_returns = clean_data.mean()
        
        self.logger.debug(f"Sample covariance calculated: {len(valid_tickers)} assets, {len(clean_data)} observations")
        
        return sample_cov, mean_returns, valid_tickers
        
    def calculate_ledoit_wolf_covariance(self, 
                                       returns_df: pd.DataFrame, 
                                       estimation_start: pd.Timestamp, 
                                       estimation_end: pd.Timestamp) -> Tuple[np.ndarray, float, pd.Series, List[str]]:
        """
        Calculate Ledoit-Wolf shrinkage covariance estimator using sklearn
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame with returns data
        estimation_start : pd.Timestamp
            Start date for estimation window
        estimation_end : pd.Timestamp
            End date for estimation window
            
        Returns:
        --------
        Tuple[np.ndarray, float, pd.Series, List[str]]
            (lw_cov, shrinkage, mean_returns, valid_tickers)
        """
        
        # Extract estimation period data
        estimation_data = returns_df.loc[estimation_start:estimation_end].copy()
        
        # Remove tickers with insufficient data
        min_obs_threshold = int(len(estimation_data) * 0.75)
        valid_tickers = []
        
        for ticker in estimation_data.columns:
            if estimation_data[ticker].notna().sum() >= min_obs_threshold:
                valid_tickers.append(ticker)
                
        if len(valid_tickers) == 0:
            raise ValueError("No tickers have sufficient data in estimation window")
            
        # Clean data
        clean_data = estimation_data[valid_tickers].ffill().dropna()
        
        if clean_data.empty:
            raise ValueError("No valid data after cleaning")
            
        # Apply Ledoit-Wolf estimator
        lw = LedoitWolf()
        lw_cov = lw.fit(clean_data.values).covariance_
        shrinkage = lw.shrinkage_
        mean_returns = clean_data.mean()
        
        self.logger.debug(f"Ledoit-Wolf covariance calculated: {len(valid_tickers)} assets, shrinkage={shrinkage:.3f}")
        
        return lw_cov, shrinkage, mean_returns, valid_tickers
        
    def solve_minimum_variance_portfolio(self, 
                                       cov_matrix: np.ndarray, 
                                       constraints: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        """
        Solve for the minimum variance portfolio using CVXPY
        
        Parameters:
        -----------
        cov_matrix : np.ndarray
            Covariance matrix
        constraints : Dict, optional
            Dictionary with weight constraints
            
        Returns:
        --------
        Tuple[Optional[np.ndarray], Optional[float], str]
            (weights, portfolio_variance, optimization_status)
        """
        
        n_assets = cov_matrix.shape[0]
        
        # Define optimization variable
        w = cp.Variable(n_assets)
        
        # Objective function: minimize portfolio variance
        # Portfolio variance = w^T * Sigma * w
        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        
        # Base constraint: weights sum to 1 (fully invested)
        base_constraints = [cp.sum(w) == 1]
        
        # Add position limits if specified
        if constraints is not None:
            if 'min_weight' in constraints and constraints['min_weight'] is not None:
                base_constraints.append(w >= constraints['min_weight'])
                
            if 'max_weight' in constraints and constraints['max_weight'] is not None:
                base_constraints.append(w <= constraints['max_weight'])
                
        # Create and solve optimization problem
        problem = cp.Problem(objective, base_constraints)
        
        # Try solvers in order of preference
        solvers_to_try = [cp.OSQP, cp.ECOS, cp.SCS]
        
        for solver in solvers_to_try:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status == cp.OPTIMAL:
                    weights = w.value
                    portfolio_variance = objective.value
                    
                    self.logger.debug(f"Optimization successful: variance={portfolio_variance:.6f}")
                    return weights, portfolio_variance, "optimal"
                elif problem.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
                    self.logger.warning(f"Solver failed: {e}")
                    return None, None, problem.status
            except Exception as e:
                self.logger.warning(f"Solver {solver.__name__} failed: {e}")
                continue
                
        self.logger.error("All solvers failed")
        return None, None, "all_solvers_failed"
        
    def create_portfolio_summary(self, 
                               weights: np.ndarray, 
                               tickers: List[str], 
                               portfolio_variance: float, 
                               shrinkage: Optional[float] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Create summary of portfolio optimization results
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        tickers : List[str]
            Ticker names
        portfolio_variance : float
            Portfolio variance
        shrinkage : float, optional
            Shrinkage parameter for Ledoit-Wolf
            
        Returns:
        --------
        Tuple[pd.DataFrame, Dict]
            (summary_df, metrics)
        """
        
        # Create portfolio positions DataFrame
        portfolio_positions = pd.DataFrame({
            'ticker': tickers,
            'weight': weights,
            'abs_weight': np.abs(weights)
        })
        
        # Sort by absolute weight (largest positions first)
        portfolio_positions = portfolio_positions.sort_values('abs_weight', ascending=False)
        
        # Calculate portfolio metrics
        portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(12)  # Annualized
        long_weight = portfolio_positions[portfolio_positions['weight'] > 0]['weight'].sum()
        short_weight = abs(portfolio_positions[portfolio_positions['weight'] < 0]['weight'].sum())
        net_exposure = long_weight - short_weight
        gross_exposure = long_weight + short_weight
        
        # Create metrics dictionary
        metrics = {
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility_annual': portfolio_volatility,
            'long_exposure': long_weight,
            'short_exposure': short_weight,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'num_positions': len([w for w in weights if abs(w) > 0.001]),  # Positions > 0.1%
            'max_position': portfolio_positions['abs_weight'].max(),
            'min_position': portfolio_positions[portfolio_positions['abs_weight'] > 0.001]['abs_weight'].min() if len(portfolio_positions[portfolio_positions['abs_weight'] > 0.001]) > 0 else 0
        }
        
        if shrinkage is not None:
            metrics['shrinkage_parameter'] = shrinkage
            
        return portfolio_positions, metrics
        
    def compare_methods(self, 
                       returns_df: pd.DataFrame, 
                       estimation_start: pd.Timestamp, 
                       estimation_end: pd.Timestamp, 
                       constraints: Optional[Dict] = None,
                       method: str = 'both') -> Dict:
        """
        Compare sample covariance vs Ledoit-Wolf shrinkage estimation
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns data
        estimation_start : pd.Timestamp
            Start date for estimation window
        estimation_end : pd.Timestamp
            End date for estimation window
        constraints : Dict, optional
            Portfolio constraints
        method : str
            Which method(s) to run: 'sample', 'ledoit_wolf', or 'both'
            
        Returns:
        --------
        Dict
            Results from both methods with weights, metrics, etc.
        """
        
        results = {}
        
        # Sample covariance method
        if method in ['sample', 'both']:
            try:
                sample_cov, mean_ret_sample, tickers_sample = self.calculate_sample_covariance(
                    returns_df, estimation_start, estimation_end
                )
                
                weights_sample, var_sample, status_sample = self.solve_minimum_variance_portfolio(
                    sample_cov, constraints
                )
                
                if weights_sample is not None:
                    portfolio_sample, metrics_sample = self.create_portfolio_summary(
                        weights_sample, tickers_sample, var_sample
                    )
                    
                    results['sample'] = {
                        'weights': weights_sample,
                        'portfolio_variance': var_sample,
                        'portfolio_positions': portfolio_sample,
                        'metrics': metrics_sample,
                        'tickers': tickers_sample,
                        'status': status_sample,
                        'mean_returns': mean_ret_sample
                    }
                else:
                    results['sample'] = {'error': f'Optimization failed: {status_sample}'}
                    
            except Exception as e:
                self.logger.error(f"Sample covariance method failed: {e}")
                results['sample'] = {'error': str(e)}
                
        # Ledoit-Wolf method
        if method in ['ledoit_wolf', 'both']:
            try:
                lw_cov, shrinkage, mean_ret_lw, tickers_lw = self.calculate_ledoit_wolf_covariance(
                    returns_df, estimation_start, estimation_end
                )
                
                weights_lw, var_lw, status_lw = self.solve_minimum_variance_portfolio(
                    lw_cov, constraints
                )
                
                if weights_lw is not None:
                    portfolio_lw, metrics_lw = self.create_portfolio_summary(
                        weights_lw, tickers_lw, var_lw, shrinkage
                    )
                    
                    results['ledoit_wolf'] = {
                        'weights': weights_lw,
                        'portfolio_variance': var_lw,
                        'portfolio_positions': portfolio_lw,
                        'metrics': metrics_lw,
                        'tickers': tickers_lw,
                        'shrinkage': shrinkage,
                        'status': status_lw,
                        'mean_returns': mean_ret_lw
                    }
                else:
                    results['ledoit_wolf'] = {'error': f'Optimization failed: {status_lw}'}
                    
            except Exception as e:
                self.logger.error(f"Ledoit-Wolf method failed: {e}")
                results['ledoit_wolf'] = {'error': str(e)}
                
        return results
        
    def calculate_efficient_frontier(self, 
                                   returns_df: pd.DataFrame,
                                   estimation_start: pd.Timestamp,
                                   estimation_end: pd.Timestamp,
                                   constraints: Optional[Dict] = None,
                                   n_points: int = 50,
                                   method: str = 'sample') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate efficient frontier points for visualization
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns data
        estimation_start : pd.Timestamp
            Start date for estimation window
        estimation_end : pd.Timestamp
            End date for estimation window
        constraints : Dict, optional
            Portfolio constraints
        n_points : int
            Number of frontier points to calculate
        method : str
            Covariance estimation method ('sample' or 'ledoit_wolf')
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (returns, volatilities) for efficient frontier points
        """
        
        # Get covariance matrix and mean returns
        if method == 'sample':
            cov_matrix, mean_returns, tickers = self.calculate_sample_covariance(
                returns_df, estimation_start, estimation_end
            )
        else:
            cov_matrix, _, mean_returns, tickers = self.calculate_ledoit_wolf_covariance(
                returns_df, estimation_start, estimation_end
            )
            
        mean_returns_array = mean_returns.values * 12  # Annualize
        n_assets = len(tickers)
        
        # Generate target returns
        min_ret = mean_returns_array.min()
        max_ret = mean_returns_array.max() * 1.5
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        # Calculate efficient frontier
        frontier_volatilities = []
        
        for target_ret in target_returns:
            try:
                # Define optimization variables
                w = cp.Variable(n_assets)
                portfolio_return = mean_returns_array.T @ w
                portfolio_variance = cp.quad_form(w, cov_matrix)
                
                # Constraints
                constraints_list = [cp.sum(w) == 1, portfolio_return == target_ret]
                
                # Add position constraints if specified
                if constraints is not None:
                    if 'min_weight' in constraints and constraints['min_weight'] is not None:
                        constraints_list.append(w >= constraints['min_weight'])
                    if 'max_weight' in constraints and constraints['max_weight'] is not None:
                        constraints_list.append(w <= constraints['max_weight'])
                
                # Solve optimization
                prob = cp.Problem(cp.Minimize(portfolio_variance), constraints_list)
                prob.solve(verbose=False)
                
                if prob.status == cp.OPTIMAL:
                    frontier_volatilities.append(np.sqrt(prob.value) * np.sqrt(12))  # Annualized vol
                else:
                    frontier_volatilities.append(np.nan)
                    
            except Exception:
                frontier_volatilities.append(np.nan)
                
        return target_returns, np.array(frontier_volatilities)
        
    def calculate_maximum_sharpe_portfolio(self,
                                         returns_df: pd.DataFrame,
                                         estimation_start: pd.Timestamp,
                                         estimation_end: pd.Timestamp,
                                         risk_free_rate: float,
                                         constraints: Optional[Dict] = None,
                                         method: str = 'sample') -> Dict:
        """
        Calculate maximum Sharpe ratio portfolio
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            Returns data
        estimation_start : pd.Timestamp
            Start date for estimation window
        estimation_end : pd.Timestamp
            End date for estimation window
        risk_free_rate : float
            Annual risk-free rate
        constraints : Dict, optional
            Portfolio constraints
        method : str
            Covariance estimation method ('sample' or 'ledoit_wolf')
            
        Returns:
        --------
        Dict
            Results with weights, Sharpe ratio, etc.
        """
        
        # Get covariance matrix and mean returns
        if method == 'sample':
            cov_matrix, mean_returns, tickers = self.calculate_sample_covariance(
                returns_df, estimation_start, estimation_end
            )
        else:
            cov_matrix, shrinkage, mean_returns, tickers = self.calculate_ledoit_wolf_covariance(
                returns_df, estimation_start, estimation_end
            )
            
        mean_returns_array = mean_returns.values * 12  # Annualize
        monthly_rf = (1 + risk_free_rate) ** (1/12) - 1
        excess_returns = mean_returns_array - risk_free_rate
        
        n_assets = len(tickers)
        
        # Define optimization variables
        w = cp.Variable(n_assets)
        portfolio_return = mean_returns_array.T @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        
        # We maximize Sharpe ratio by minimizing the inverse
        # Sharpe = (return - rf) / volatility
        # We minimize volatility / (return - rf), which is equivalent
        objective = cp.Minimize(portfolio_variance / (portfolio_return - risk_free_rate))
        
        # Constraints
        constraints_list = [cp.sum(w) == 1, portfolio_return >= risk_free_rate + 0.001]  # Ensure positive excess return
        
        # Add position constraints if specified
        if constraints is not None:
            if 'min_weight' in constraints and constraints['min_weight'] is not None:
                constraints_list.append(w >= constraints['min_weight'])
            if 'max_weight' in constraints and constraints['max_weight'] is not None:
                constraints_list.append(w <= constraints['max_weight'])
                
        try:
            # Solve optimization
            prob = cp.Problem(objective, constraints_list)
            prob.solve(verbose=False)
            
            if prob.status == cp.OPTIMAL:
                weights = w.value
                portfolio_ret = float(portfolio_return.value)
                portfolio_vol = np.sqrt(portfolio_variance.value) * np.sqrt(12)
                sharpe_ratio = (portfolio_ret - risk_free_rate) / portfolio_vol
                
                # Create summary
                portfolio_positions, metrics = self.create_portfolio_summary(
                    weights, tickers, portfolio_variance.value,
                    shrinkage if method == 'ledoit_wolf' else None
                )
                
                result = {
                    'weights': weights,
                    'portfolio_return': portfolio_ret,
                    'portfolio_volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'portfolio_positions': portfolio_positions,
                    'metrics': metrics,
                    'tickers': tickers,
                    'status': 'optimal'
                }
                
                if method == 'ledoit_wolf':
                    result['shrinkage'] = shrinkage
                    
                return result
            else:
                return {'error': f'Optimization failed: {prob.status}'}
                
        except Exception as e:
            self.logger.error(f"Maximum Sharpe optimization failed: {e}")
            return {'error': str(e)}