# src/portfolio/metrics.py
"""
Performance analysis and metrics calculation for portfolio optimization
Handles return analysis, risk metrics, and diagnostic reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for portfolio optimization results
    Calculates risk metrics, performance statistics, and diagnostic insights
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_performance_metrics(self, 
                                    backtest_results: pd.DataFrame, 
                                    risk_free_rate: float) -> Dict:
        """
        Calculate comprehensive performance metrics for backtest results
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            DataFrame with backtest results from rolling window analysis
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio calculation
            
        Returns:
        --------
        Dict
            Performance metrics for both methods
        """
        
        metrics = {}
        
        # Calculate metrics for both sample and Ledoit-Wolf methods
        for method in ['sample', 'lw']:
            return_col = f'{method}_return'
            
            if return_col not in backtest_results.columns:
                continue
                
            returns = backtest_results[return_col].dropna()
            
            if len(returns) == 0:
                continue
                
            # Basic return statistics
            total_periods = len(returns)
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (12 / total_periods) - 1
            monthly_volatility = returns.std()
            annualized_volatility = monthly_volatility * np.sqrt(12)
            
            # Risk-adjusted metrics
            monthly_rf = (1 + risk_free_rate) ** (1/12) - 1
            excess_returns = returns - monthly_rf
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(12) if excess_returns.std() > 0 else 0
            
            # Downside metrics
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(12) if len(negative_returns) > 0 else 0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate average time to recovery from drawdowns
            drawdown_recovery_times = self._calculate_drawdown_recovery_times(cumulative_returns)
            avg_recovery_time = np.mean(drawdown_recovery_times) if drawdown_recovery_times else 0
            
            # Win/Loss statistics
            positive_months = (returns > 0).sum()
            negative_months = (returns < 0).sum()
            win_rate = positive_months / total_periods if total_periods > 0 else 0
            
            avg_win = returns[returns > 0].mean() if positive_months > 0 else 0
            avg_loss = returns[returns < 0].mean() if negative_months > 0 else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Statistical tests
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            jarque_bera_stat, jarque_bera_p = stats.jarque_bera(returns)
            
            # Value at Risk (VaR) and Conditional VaR
            var_95 = returns.quantile(0.05)  # 5% VaR
            var_99 = returns.quantile(0.01)  # 1% VaR
            cvar_95 = returns[returns <= var_95].mean()  # Expected shortfall
            cvar_99 = returns[returns <= var_99].mean()
            
            # Calmar ratio (annualized return / max drawdown)
            calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
            
            # Store all metrics
            metrics[method] = {
                'total_periods': total_periods,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'avg_recovery_time': avg_recovery_time,
                'win_rate': win_rate,
                'avg_monthly_return': returns.mean(),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'jarque_bera_p': jarque_bera_p,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'best_month': returns.max(),
                'worst_month': returns.min(),
                'positive_months': positive_months,
                'negative_months': negative_months,
                'monthly_volatility': monthly_volatility
            }
            
        return metrics
        
    def calculate_turnover_metrics(self, 
                                 portfolio_weights: pd.DataFrame) -> Dict:
        """
        Calculate portfolio turnover metrics for both methods
        
        Parameters:
        -----------
        portfolio_weights : pd.DataFrame
            DataFrame with portfolio weights over time
            
        Returns:
        --------
        Dict
            Turnover statistics for both methods
        """
        
        turnover_metrics = {}
        
        for method in ['sample', 'lw']:
            turnover_series = self._calculate_portfolio_turnover(portfolio_weights, method)
            
            if len(turnover_series) == 0:
                continue
                
            # Calculate turnover statistics
            turnover_metrics[method] = {
                'mean_turnover': turnover_series.mean(),
                'median_turnover': turnover_series.median(),
                'std_turnover': turnover_series.std(),
                'min_turnover': turnover_series.min(),
                'max_turnover': turnover_series.max(),
                'turnover_25th': turnover_series.quantile(0.25),
                'turnover_75th': turnover_series.quantile(0.75),
                'turnover_series': turnover_series  # Store for plotting
            }
            
        return turnover_metrics
        
    def calculate_stability_metrics(self, 
                                  portfolio_weights: pd.DataFrame) -> Dict:
        """
        Calculate portfolio weight stability metrics
        
        Parameters:
        -----------
        portfolio_weights : pd.DataFrame
            DataFrame with portfolio weights over time
            
        Returns:
        --------
        Dict
            Stability metrics for both methods
        """
        
        stability_metrics = {}
        
        for method in ['sample', 'lw']:
            stability_series = self._calculate_weight_stability(portfolio_weights, method)
            
            if len(stability_series) == 0:
                continue
                
            # Calculate stability statistics
            stability_metrics[method] = {
                'mean_stability': stability_series.mean(),
                'median_stability': stability_series.median(),
                'std_stability': stability_series.std(),
                'min_stability': stability_series.min(),
                'max_stability': stability_series.max(),
                'stability_trend': np.polyfit(range(len(stability_series)), stability_series.values, 1)[0],
                'stability_series': stability_series  # Store for plotting
            }
            
        return stability_metrics
        
    def calculate_weight_distribution_metrics(self, 
                                            portfolio_weights: pd.DataFrame) -> Dict:
        """
        Calculate weight distribution metrics showing concentration patterns
        
        Parameters:
        -----------
        portfolio_weights : pd.DataFrame
            DataFrame with portfolio weights over time
            
        Returns:
        --------
        Dict
            Weight distribution statistics
        """
        
        distribution_metrics = {}
        
        for method in ['sample', 'lw']:
            # Select weight columns for the specified method
            method_suffix = '_sample' if method == 'sample' else '_lw'
            weight_cols = [col for col in portfolio_weights.columns if col.endswith(method_suffix)]
            
            if len(weight_cols) == 0:
                continue
                
            # Extract weight matrix
            weight_matrix = portfolio_weights[weight_cols].fillna(0)
            
            # Calculate cross-sectional standard deviation of weights at each date
            cross_sectional_stds = weight_matrix.std(axis=1)
            
            # Calculate Herfindahl-Hirschman Index over time
            hhi_series = (weight_matrix ** 2).sum(axis=1)
            
            # Calculate effective number of positions over time
            effective_positions = 1 / hhi_series
            
            # Top N concentration metrics
            top_3_concentration = weight_matrix.apply(lambda row: row.nlargest(3).sum(), axis=1)
            top_5_concentration = weight_matrix.apply(lambda row: row.nlargest(5).sum(), axis=1)
            
            distribution_metrics[method] = {
                'mean_weight_std': cross_sectional_stds.mean(),
                'median_weight_std': cross_sectional_stds.median(),
                'max_weight_std': cross_sectional_stds.max(),
                'min_weight_std': cross_sectional_stds.min(),
                'mean_hhi': hhi_series.mean(),
                'mean_effective_positions': effective_positions.mean(),
                'mean_top_3_concentration': top_3_concentration.mean(),
                'mean_top_5_concentration': top_5_concentration.mean(),
                'weight_std_series': cross_sectional_stds,
                'hhi_series': hhi_series,
                'effective_positions_series': effective_positions
            }
            
        return distribution_metrics
        
    def create_performance_summary_table(self, 
                                        performance_metrics: Dict) -> pd.DataFrame:
        """
        Create formatted summary table of performance metrics
        
        Parameters:
        -----------
        performance_metrics : Dict
            Performance metrics from calculate_performance_metrics
            
        Returns:
        --------
        pd.DataFrame
            Formatted performance comparison table
        """
        
        if not performance_metrics:
            return pd.DataFrame()
            
        # Define metrics to display with formatting
        display_metrics = [
            ('Total Return', 'total_return', '{:.2%}'),
            ('Annualized Return', 'annualized_return', '{:.2%}'),
            ('Annualized Volatility', 'annualized_volatility', '{:.2%}'),
            ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
            ('Sortino Ratio', 'sortino_ratio', '{:.3f}'),
            ('Calmar Ratio', 'calmar_ratio', '{:.3f}'),
            ('Maximum Drawdown', 'max_drawdown', '{:.2%}'),
            ('Win Rate', 'win_rate', '{:.1%}'),
            ('Best Month', 'best_month', '{:.2%}'),
            ('Worst Month', 'worst_month', '{:.2%}'),
            ('VaR 95%', 'var_95', '{:.2%}'),
            ('CVaR 95%', 'cvar_95', '{:.2%}'),
            ('Skewness', 'skewness', '{:.3f}'),
            ('Kurtosis', 'kurtosis', '{:.3f}')
        ]
        
        # Build summary table
        summary_data = {}
        
        for method in performance_metrics:
            method_name = 'Sample Covariance' if method == 'sample' else 'Ledoit-Wolf'
            summary_data[method_name] = []
            
            for display_name, metric_key, format_str in display_metrics:
                value = performance_metrics[method].get(metric_key, np.nan)
                if not np.isnan(value):
                    formatted_value = format_str.format(value)
                else:
                    formatted_value = 'N/A'
                summary_data[method_name].append(formatted_value)
                
        # Create DataFrame
        metric_names = [item[0] for item in display_metrics]
        summary_df = pd.DataFrame(summary_data, index=metric_names)
        
        return summary_df
        
    def create_diagnostic_report(self, 
                               performance_metrics: Dict, 
                               turnover_metrics: Dict, 
                               stability_metrics: Dict) -> str:
        """
        Create comprehensive diagnostic report with insights
        
        Parameters:
        -----------
        performance_metrics : Dict
            Performance analysis results
        turnover_metrics : Dict
            Turnover analysis results
        stability_metrics : Dict
            Stability analysis results
            
        Returns:
        --------
        str
            Formatted diagnostic report
        """
        
        report_sections = []
        
        # Performance comparison
        if 'sample' in performance_metrics and 'lw' in performance_metrics:
            sample_sharpe = performance_metrics['sample']['sharpe_ratio']
            lw_sharpe = performance_metrics['lw']['sharpe_ratio']
            sample_vol = performance_metrics['sample']['annualized_volatility']
            lw_vol = performance_metrics['lw']['annualized_volatility']
            
            report_sections.append("PERFORMANCE COMPARISON")
            report_sections.append("=" * 50)
            
            # Sharpe ratio comparison
            if lw_sharpe > sample_sharpe:
                improvement = (lw_sharpe - sample_sharpe) / sample_sharpe * 100
                report_sections.append(f"• Ledoit-Wolf shows superior risk-adjusted returns (Sharpe: {lw_sharpe:.3f} vs {sample_sharpe:.3f}, +{improvement:.1f}%)")
            else:
                improvement = (sample_sharpe - lw_sharpe) / lw_sharpe * 100 if lw_sharpe != 0 else 0
                report_sections.append(f"• Sample covariance shows superior risk-adjusted returns (Sharpe: {sample_sharpe:.3f} vs {lw_sharpe:.3f}, +{improvement:.1f}%)")
                
            # Volatility comparison
            if lw_vol < sample_vol:
                vol_reduction = (sample_vol - lw_vol) / sample_vol * 100
                report_sections.append(f"• Ledoit-Wolf reduces portfolio volatility by {vol_reduction:.1f}%")
            else:
                vol_increase = (lw_vol - sample_vol) / sample_vol * 100
                report_sections.append(f"• Sample covariance shows {vol_increase:.1f}% lower volatility")
                
            # Risk-adjusted performance assessment
            sample_calmar = performance_metrics['sample']['calmar_ratio']
            lw_calmar = performance_metrics['lw']['calmar_ratio']
            report_sections.append(f"• Calmar ratios: Sample {sample_calmar:.2f}, Ledoit-Wolf {lw_calmar:.2f}")
            
        # Turnover analysis
        if turnover_metrics:
            report_sections.append("\nTURNOVER ANALYSIS")
            report_sections.append("=" * 50)
            
            for method in turnover_metrics:
                method_name = 'Sample' if method == 'sample' else 'Ledoit-Wolf'
                mean_turnover = turnover_metrics[method]['mean_turnover']
                max_turnover = turnover_metrics[method]['max_turnover']
                
                report_sections.append(f"• {method_name} average monthly turnover: {mean_turnover:.1%}")
                report_sections.append(f"  Maximum monthly turnover: {max_turnover:.1%}")
                
                # Turnover assessment
                if mean_turnover > 0.5:
                    report_sections.append(f"  WARNING: High turnover may indicate unstable optimization or high transaction costs")
                elif mean_turnover < 0.1:
                    report_sections.append(f"  NOTE: Low turnover suggests stable portfolio weights")
                    
        # Stability analysis
        if stability_metrics:
            report_sections.append("\nSTABILITY ANALYSIS")
            report_sections.append("=" * 50)
            
            for method in stability_metrics:
                method_name = 'Sample' if method == 'sample' else 'Ledoit-Wolf'
                trend = stability_metrics[method]['stability_trend']
                mean_stability = stability_metrics[method]['mean_stability']
                
                report_sections.append(f"• {method_name} average weight dispersion: {mean_stability:.3f}")
                
                # Trend analysis
                if abs(trend) < 0.001:
                    report_sections.append(f"  Weight stability remains consistent over time")
                elif trend > 0.001:
                    report_sections.append(f"  Portfolio weights becoming less stable over time (trend: {trend:.4f})")
                else:
                    report_sections.append(f"  Portfolio weights becoming more stable over time (trend: {trend:.4f})")
                    
        return '\n'.join(report_sections)
        
    def calculate_comprehensive_metrics(self, 
                                      backtest_results: pd.DataFrame,
                                      portfolio_weights: pd.DataFrame,
                                      risk_free_rate: float) -> Dict:
        """
        Calculate all performance metrics in one comprehensive analysis
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            Backtest results from rolling window analysis
        portfolio_weights : pd.DataFrame
            Portfolio weights over time
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        Dict
            Comprehensive analysis results
        """
        
        self.logger.info("Calculating comprehensive performance metrics...")
        
        # Calculate all metric types
        performance_metrics = self.calculate_performance_metrics(backtest_results, risk_free_rate)
        turnover_metrics = self.calculate_turnover_metrics(portfolio_weights)
        stability_metrics = self.calculate_stability_metrics(portfolio_weights)
        distribution_metrics = self.calculate_weight_distribution_metrics(portfolio_weights)
        
        # Create summary table
        summary_table = self.create_performance_summary_table(performance_metrics)
        
        # Create diagnostic report
        diagnostic_report = self.create_diagnostic_report(
            performance_metrics, turnover_metrics, stability_metrics
        )
        
        return {
            'performance_metrics': performance_metrics,
            'turnover_metrics': turnover_metrics,
            'stability_metrics': stability_metrics,
            'distribution_metrics': distribution_metrics,
            'summary_table': summary_table,
            'diagnostic_report': diagnostic_report
        }
        
    def _calculate_portfolio_turnover(self, 
                                    weights_df: pd.DataFrame, 
                                    method: str = 'sample') -> pd.Series:
        """Calculate monthly portfolio turnover based on weight changes"""
        
        # Select weight columns for the specified method
        method_suffix = '_sample' if method == 'sample' else '_lw'
        weight_cols = [col for col in weights_df.columns if col.endswith(method_suffix)]
        
        if len(weight_cols) == 0:
            return pd.Series(dtype=float)
            
        # Extract weight matrix
        weight_matrix = weights_df[weight_cols].fillna(0)
        
        # Calculate turnover as sum of absolute weight changes
        turnover_list = []
        dates = weight_matrix.index
        
        for i in range(1, len(weight_matrix)):
            current_weights = weight_matrix.iloc[i].values
            previous_weights = weight_matrix.iloc[i-1].values
            
            # Turnover = sum of absolute weight changes
            turnover = np.sum(np.abs(current_weights - previous_weights))
            turnover_list.append(turnover)
            
        # Create turnover series
        turnover_series = pd.Series(turnover_list, index=dates[1:])
        
        return turnover_series
        
    def _calculate_weight_stability(self, 
                                  weights_df: pd.DataFrame, 
                                  method: str = 'sample', 
                                  window: int = 12) -> pd.Series:
        """Calculate portfolio weight stability using rolling cross-sectional standard deviation"""
        
        # Select weight columns for the specified method
        method_suffix = '_sample' if method == 'sample' else '_lw'
        weight_cols = [col for col in weights_df.columns if col.endswith(method_suffix)]
        
        if len(weight_cols) == 0:
            return pd.Series(dtype=float)
            
        # Extract weight matrix
        weight_matrix = weights_df[weight_cols].fillna(0)
        
        # Calculate cross-sectional standard deviation of weights at each date
        cross_sectional_std = weight_matrix.std(axis=1)
        
        # Calculate rolling average of cross-sectional standard deviation
        stability_series = cross_sectional_std.rolling(window=window, min_periods=1).mean()
        
        return stability_series
        
    def _calculate_drawdown_recovery_times(self, 
                                         cumulative_returns: pd.Series) -> List[int]:
        """Calculate recovery times for drawdown periods"""
        
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                drawdown_start = i
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_times.append(i - drawdown_start)
                    
        return recovery_times