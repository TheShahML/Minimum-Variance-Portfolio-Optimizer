# src/utils/plotting.py
"""
Plotting utilities for portfolio optimization visualization
Creates publication-quality charts for performance analysis and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

# Set default plotting style
plt.style.use('default')
sns.set_palette("husl")

class PortfolioPlotter:
    """
    Creates visualizations for portfolio optimization results
    Handles performance charts, risk analysis, and diagnostic plots
    """
    
    def __init__(self, style: str = 'modern', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter with style preferences
        
        Parameters:
        -----------
        style : str
            Plotting style ('modern', 'classic', 'minimal')
        figsize : Tuple[int, int]
            Default figure size
        """
        
        self.style = style
        self.default_figsize = figsize
        self.logger = logging.getLogger(__name__)
        
        # Color schemes for different styles
        self.color_schemes = {
            'modern': {
                'sample': '#2E86AB',     # Blue
                'ledoit_wolf': '#A23B72',  # Purple
                'benchmark': '#F18F01',     # Orange
                'risk_free': '#C73E1D',     # Red
                'background': '#F8F9FA',
                'grid': '#E9ECEF'
            },
            'classic': {
                'sample': 'blue',
                'ledoit_wolf': 'red', 
                'benchmark': 'green',
                'risk_free': 'black',
                'background': 'white',
                'grid': 'lightgray'
            }
        }
        
        self._set_plot_style()
        
    def plot_cumulative_returns(self, 
                               backtest_results: pd.DataFrame,
                               title: str = "Cumulative Returns Comparison",
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot cumulative returns for both optimization methods
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            Backtest results with return columns
        title : str
            Chart title
        figsize : Tuple[int, int], optional
            Figure size override
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = self.color_schemes[self.style]
        
        # Plot cumulative returns for both methods
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    cumulative = (1 + returns).cumprod()
                    ax.plot(cumulative.index, cumulative.values,
                           color=colors[color_key], linewidth=2.5, 
                           label=label, alpha=0.9)
                    
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_ylabel('Cumulative Return', fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, color=colors['grid'])
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{(y-1):.1%}'))
        
        plt.tight_layout()
        return fig
        
    def plot_rolling_metrics(self, 
                            backtest_results: pd.DataFrame,
                            figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot rolling Sharpe ratio and volatility
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            Backtest results
        figsize : Tuple[int, int], optional
            Figure size override
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object with subplots
        """
        
        figsize = figsize or (14, 10)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Rolling Performance Metrics', fontsize=16, fontweight='bold')
        
        colors = self.color_schemes[self.style]
        
        # Plot 1: Rolling Sharpe Ratio
        ax1 = axes[0, 0]
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    rolling_sharpe = returns.rolling(12).apply(
                        lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
                    )
                    ax1.plot(rolling_sharpe.index, rolling_sharpe.values,
                            color=colors[color_key], linewidth=2, label=label, alpha=0.8)
                    
        ax1.set_title('12-Month Rolling Sharpe Ratio', fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 2: Rolling Volatility
        ax2 = axes[0, 1]
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    rolling_vol = returns.rolling(12).std() * np.sqrt(12)
                    ax2.plot(rolling_vol.index, rolling_vol.values,
                            color=colors[color_key], linewidth=2, label=label, alpha=0.8)
                    
        ax2.set_title('12-Month Rolling Volatility', fontweight='bold')
        ax2.set_ylabel('Annualized Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot 3: Return Distribution
        ax3 = axes[1, 0]
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    ax3.hist(returns.values, bins=30, alpha=0.6, 
                            color=colors[color_key], label=label, density=True)
                    
        ax3.set_title('Return Distribution', fontweight='bold')
        ax3.set_xlabel('Monthly Return')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # Plot 4: Ledoit-Wolf Shrinkage Evolution
        ax4 = axes[1, 1]
        if 'lw_shrinkage' in backtest_results.columns:
            shrinkage_data = backtest_results['lw_shrinkage'].dropna()
            ax4.plot(shrinkage_data.index, shrinkage_data.values,
                    color=colors['ledoit_wolf'], linewidth=2, alpha=0.8)
            ax4.fill_between(shrinkage_data.index, shrinkage_data.values, 
                           alpha=0.3, color=colors['ledoit_wolf'])
        else:
            ax4.text(0.5, 0.5, 'Shrinkage data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            
        ax4.set_title('Ledoit-Wolf Shrinkage Parameter', fontweight='bold')
        ax4.set_ylabel('Shrinkage Intensity')
        ax4.set_xlabel('Date')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_turnover_analysis(self, 
                              portfolio_weights: pd.DataFrame,
                              figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot portfolio turnover analysis
        
        Parameters:
        -----------
        portfolio_weights : pd.DataFrame
            Portfolio weights over time
        figsize : Tuple[int, int], optional
            Figure size override
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        
        figsize = figsize or (14, 6)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Portfolio Turnover Analysis', fontsize=16, fontweight='bold')
        
        colors = self.color_schemes[self.style]
        
        # Calculate turnover for both methods
        turnover_data = {}
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            turnover_series = self._calculate_turnover(portfolio_weights, method)
            if len(turnover_series) > 0:
                turnover_data[label] = {
                    'series': turnover_series,
                    'color': colors[color_key]
                }
                
        # Plot 1: Turnover Time Series
        ax1 = axes[0]
        for label, data in turnover_data.items():
            ax1.plot(data['series'].index, data['series'].values,
                    color=data['color'], linewidth=2, label=label, alpha=0.8)
            
        ax1.set_title('Monthly Portfolio Turnover', fontweight='bold')
        ax1.set_ylabel('Turnover')
        ax1.set_xlabel('Date')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot 2: Turnover Distribution
        ax2 = axes[1]
        turnover_values = []
        labels = []
        colors_list = []
        
        for label, data in turnover_data.items():
            turnover_values.append(data['series'].values)
            labels.append(label)
            colors_list.append(data['color'])
            
        if turnover_values:
            box_plot = ax2.boxplot(turnover_values, labels=labels, patch_artist=True)
            for patch, color in zip(box_plot['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
        ax2.set_title('Turnover Distribution', fontweight='bold')
        ax2.set_ylabel('Turnover')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        return fig
        
    def plot_weight_evolution(self, 
                             portfolio_weights: pd.DataFrame,
                             top_n: int = 5,
                             method: str = 'sample',
                             figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot evolution of top portfolio holdings over time
        
        Parameters:
        -----------
        portfolio_weights : pd.DataFrame
            Portfolio weights over time
        top_n : int
            Number of top holdings to display
        method : str
            Method to plot ('sample' or 'lw')
        figsize : Tuple[int, int], optional
            Figure size override
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        method_suffix = '_sample' if method == 'sample' else '_lw'
        weight_cols = [col for col in portfolio_weights.columns if col.endswith(method_suffix)]
        
        if not weight_cols:
            ax.text(0.5, 0.5, f'No weight data available for {method} method',
                   ha='center', va='center', transform=ax.transAxes)
            return fig
            
        # Find top holdings by average absolute weight
        avg_weights = portfolio_weights[weight_cols].abs().mean().sort_values(ascending=False)
        top_holdings = avg_weights.head(top_n).index
        
        # Plot evolution of top holdings
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_holdings)))
        
        for i, col in enumerate(top_holdings):
            ticker = col.replace(method_suffix, '')
            weights_ts = portfolio_weights[col].fillna(0)
            ax.plot(weights_ts.index, weights_ts.values,
                   linewidth=2, label=ticker, alpha=0.8, color=colors[i])
            
        method_name = 'Sample Covariance' if method == 'sample' else 'Ledoit-Wolf'
        ax.set_title(f'Top {top_n} Holdings Weight Evolution ({method_name})', 
                    fontweight='bold', fontsize=14)
        ax.set_ylabel('Portfolio Weight')
        ax.set_xlabel('Date')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        return fig
        
    def plot_efficient_frontier(self, 
                               frontier_data: Dict,
                               portfolio_points: Optional[Dict] = None,
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot efficient frontier with portfolio points
        
        Parameters:
        -----------
        frontier_data : Dict
            Efficient frontier data with 'returns' and 'volatilities'
        portfolio_points : Dict, optional
            Portfolio points to highlight on frontier
        figsize : Tuple[int, int], optional
            Figure size override
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = self.color_schemes[self.style]
        
        # Plot efficient frontier
        if 'sample' in frontier_data:
            sample_data = frontier_data['sample']
            valid_mask = ~np.isnan(sample_data['volatilities'])
            ax.plot(sample_data['volatilities'][valid_mask], 
                   sample_data['returns'][valid_mask],
                   'b-', linewidth=2, alpha=0.8, 
                   label='Sample Covariance Frontier', 
                   color=colors['sample'])
                   
        if 'ledoit_wolf' in frontier_data:
            lw_data = frontier_data['ledoit_wolf']
            valid_mask = ~np.isnan(lw_data['volatilities'])
            ax.plot(lw_data['volatilities'][valid_mask], 
                   lw_data['returns'][valid_mask],
                   'r-', linewidth=2, alpha=0.8, 
                   label='Ledoit-Wolf Frontier',
                   color=colors['ledoit_wolf'])
                   
        # Plot portfolio points if provided
        if portfolio_points:
            for point_name, point_data in portfolio_points.items():
                ax.scatter(point_data['volatility'], point_data['return'],
                          s=100, alpha=0.8, label=point_name,
                          edgecolors='white', linewidth=2)
                          
        ax.set_xlabel('Annualized Volatility', fontweight='bold')
        ax.set_ylabel('Annualized Expected Return', fontweight='bold')
        ax.set_title('Efficient Frontier', fontweight='bold', fontsize=16)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format axes as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        return fig
        
    def plot_risk_metrics_comparison(self, 
                                   performance_metrics: Dict,
                                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot comparison of risk metrics between methods
        
        Parameters:
        -----------
        performance_metrics : Dict
            Performance metrics for both methods
        figsize : Tuple[int, int], optional
            Figure size override
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        
        figsize = figsize or (12, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Risk Metrics Comparison', fontsize=16, fontweight='bold')
        
        colors = self.color_schemes[self.style]
        
        # Prepare data
        methods = ['Sample Covariance', 'Ledoit-Wolf']
        sample_metrics = performance_metrics.get('sample', {})
        lw_metrics = performance_metrics.get('lw', {})
        
        # Plot 1: Sharpe vs Sortino Ratios
        ax1 = axes[0, 0]
        sharpe_ratios = [sample_metrics.get('sharpe_ratio', 0), lw_metrics.get('sharpe_ratio', 0)]
        sortino_ratios = [sample_metrics.get('sortino_ratio', 0), lw_metrics.get('sortino_ratio', 0)]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax1.bar(x - width/2, sharpe_ratios, width, label='Sharpe Ratio', 
               color=colors['sample'], alpha=0.7)
        ax1.bar(x + width/2, sortino_ratios, width, label='Sortino Ratio', 
               color=colors['ledoit_wolf'], alpha=0.7)
               
        ax1.set_title('Risk-Adjusted Returns', fontweight='bold')
        ax1.set_ylabel('Ratio')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Volatility vs Max Drawdown
        ax2 = axes[0, 1]
        volatilities = [sample_metrics.get('annualized_volatility', 0), 
                       lw_metrics.get('annualized_volatility', 0)]
        max_drawdowns = [abs(sample_metrics.get('max_drawdown', 0)), 
                        abs(lw_metrics.get('max_drawdown', 0))]
        
        ax2.bar(x - width/2, volatilities, width, label='Volatility', 
               color=colors['sample'], alpha=0.7)
        ax2.bar(x + width/2, max_drawdowns, width, label='Max Drawdown', 
               color=colors['ledoit_wolf'], alpha=0.7)
               
        ax2.set_title('Risk Measures', fontweight='bold')
        ax2.set_ylabel('Percentage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot 3: VaR Comparison
        ax3 = axes[1, 0]
        var95_values = [abs(sample_metrics.get('var_95', 0)), 
                       abs(lw_metrics.get('var_95', 0))]
        cvar95_values = [abs(sample_metrics.get('cvar_95', 0)), 
                        abs(lw_metrics.get('cvar_95', 0))]
        
        ax3.bar(x - width/2, var95_values, width, label='VaR 95%', 
               color=colors['sample'], alpha=0.7)
        ax3.bar(x + width/2, cvar95_values, width, label='CVaR 95%', 
               color=colors['ledoit_wolf'], alpha=0.7)
               
        ax3.set_title('Tail Risk Measures', fontweight='bold')
        ax3.set_ylabel('Monthly Loss')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot 4: Distribution Properties
        ax4 = axes[1, 1]
        skewness_values = [sample_metrics.get('skewness', 0), 
                          lw_metrics.get('skewness', 0)]
        kurtosis_values = [sample_metrics.get('kurtosis', 0), 
                          lw_metrics.get('kurtosis', 0)]
        
        ax4.bar(x - width/2, skewness_values, width, label='Skewness', 
               color=colors['sample'], alpha=0.7)
        ax4.bar(x + width/2, kurtosis_values, width, label='Excess Kurtosis', 
               color=colors['ledoit_wolf'], alpha=0.7)
               
        ax4.set_title('Distribution Properties', fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def create_performance_dashboard(self, 
                                   backtest_results: pd.DataFrame,
                                   portfolio_weights: pd.DataFrame,
                                   performance_metrics: Dict,
                                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create original performance dashboard with turnover distribution replacing HHI
        """
        
        figsize = figsize or (20, 12)
        fig = plt.figure(figsize=figsize)
        fig.suptitle('Portfolio Optimization Performance Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # Create grid layout - same as original
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        colors = self.color_schemes[self.style]
        
        # 1. Cumulative Returns (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    cumulative = (1 + returns).cumprod()
                    ax1.plot(cumulative.index, cumulative.values,
                           color=colors[color_key], linewidth=3, 
                           label=label, alpha=0.9)
                    
        ax1.set_title('Cumulative Returns', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{(y-1):.0%}'))
        
        # 2. Key Metrics Table (spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.axis('off')
        
        table_data = []
        metrics_to_show = [
            ('Sharpe Ratio', 'sharpe_ratio', '.3f'),
            ('Annual Return', 'annualized_return', '.1%'),
            ('Annual Vol', 'annualized_volatility', '.1%'),
            ('Max Drawdown', 'max_drawdown', '.1%')
        ]
        
        for metric_name, key, fmt in metrics_to_show:
            sample_val = performance_metrics.get('sample', {}).get(key, 0)
            lw_val = performance_metrics.get('lw', {}).get(key, 0)
            table_data.append([metric_name, f"{sample_val:{fmt}}", f"{lw_val:{fmt}}"])
            
        table = ax2.table(cellText=table_data,
                         colLabels=['Metric', 'Sample Cov', 'Ledoit-Wolf'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax2.set_title('Key Performance Metrics', fontweight='bold', fontsize=14)
        
        # 3. Rolling Sharpe (spans 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    rolling_sharpe = returns.rolling(12).apply(
                        lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
                    )
                    ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                            color=colors[color_key], linewidth=2, label=label, alpha=0.8)
                    
        ax3.set_title('12-Month Rolling Sharpe Ratio', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Turnover Analysis (spans 2 columns)
        ax4 = fig.add_subplot(gs[1, 2:])
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            turnover_series = self._calculate_turnover(portfolio_weights, method)
            if len(turnover_series) > 0:
                ax4.plot(turnover_series.index, turnover_series.values,
                        color=colors[color_key], linewidth=2, label=label, alpha=0.8)
                
        ax4.set_title('Portfolio Turnover', fontweight='bold', fontsize=14)
        ax4.set_ylabel('Monthly Turnover')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 5. Return Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    ax5.hist(returns.values, bins=20, alpha=0.6, 
                            color=colors[color_key], label=label, density=True)
                    
        ax5.set_title('Return Distribution', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Monthly Return')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Drawdown Analysis
        ax6 = fig.add_subplot(gs[2, 1])
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    ax6.fill_between(drawdown.index, drawdown.values, 0,
                                   color=colors[color_key], alpha=0.6, label=label)
                    
        ax6.set_title('Drawdown Evolution', fontweight='bold', fontsize=12)
        ax6.set_ylabel('Drawdown')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 7. TURNOVER DISTRIBUTION (replaces HHI)
        ax7 = fig.add_subplot(gs[2, 2])
        turnover_data = []
        labels = []
        colors_list = []
        
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            turnover_series = self._calculate_turnover(portfolio_weights, method)
            if len(turnover_series) > 0:
                turnover_data.append(turnover_series.values)
                labels.append(label)
                colors_list.append(colors[color_key])
                
        if turnover_data:
            box_plot = ax7.boxplot(turnover_data, labels=labels, patch_artist=True)
            for patch, color in zip(box_plot['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
        ax7.set_title('Turnover Distribution', fontweight='bold', fontsize=12)
        ax7.set_ylabel('Monthly Turnover')
        ax7.grid(True, alpha=0.3)
        ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 8. Shrinkage Evolution
        ax8 = fig.add_subplot(gs[2, 3])
        if 'lw_shrinkage' in backtest_results.columns:
            shrinkage_data = backtest_results['lw_shrinkage'].dropna()
            ax8.plot(shrinkage_data.index, shrinkage_data.values,
                    color=colors['ledoit_wolf'], linewidth=2, alpha=0.8)
            ax8.fill_between(shrinkage_data.index, shrinkage_data.values, 
                           alpha=0.3, color=colors['ledoit_wolf'])
        else:
            ax8.text(0.5, 0.5, 'Shrinkage data\nnot available', 
                    ha='center', va='center', transform=ax8.transAxes)
            
        ax8.set_title('LW Shrinkage Parameter', fontweight='bold', fontsize=12)
        ax8.set_ylabel('Shrinkage')
        ax8.grid(True, alpha=0.3)
        
        return fig
        
    def plot_efficient_frontier_comparison(self, 
                                         returns_data: pd.DataFrame,
                                         config: Dict,
                                         figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        NEW: Plot efficient frontier comparing both methods with CAL
        """
        
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = self.color_schemes[self.style]
        
        try:
            final_tickers = config.get('final_tickers', config.get('tickers', []))
            estimation_window = config.get('estimation_window', 36)
            risk_free_rate = config.get('risk_free_rate', 0.042)
            constraints_dict = config.get('constraints', {})
            
            # Use last estimation window
            all_dates = returns_data.index.sort_values()
            if len(all_dates) >= estimation_window:
                end_date = all_dates[-1]
                start_idx = len(all_dates) - estimation_window
                start_date = all_dates[start_idx]
                
                final_period_data = returns_data[final_tickers].loc[start_date:end_date].ffill().dropna()
                
                if len(final_period_data) >= 12:
                    # Calculate mean returns (annualized)
                    mean_returns = final_period_data.mean().values * 12
                    
                    # Sample covariance
                    sample_cov = final_period_data.cov().values
                    
                    # Ledoit-Wolf covariance
                    from sklearn.covariance import LedoitWolf
                    lw = LedoitWolf()
                    lw_cov = lw.fit(final_period_data.values).covariance_
                    
                    # Generate efficient frontier for both methods (only upper half)
                    min_ret = max(mean_returns.min(), risk_free_rate)  # Start from risk-free rate or min return
                    target_returns = np.linspace(min_ret, mean_returns.max() * 1.5, 30)
                    
                    sample_vols = []
                    lw_vols = []
                    
                    for target_ret in target_returns:
                        # Sample covariance frontier
                        try:
                            import cvxpy as cp
                            w = cp.Variable(len(final_tickers))
                            portfolio_return = mean_returns.T @ w
                            portfolio_variance = cp.quad_form(w, sample_cov)
                            
                            constraints = [cp.sum(w) == 1, portfolio_return == target_ret]
                            
                            if constraints_dict.get('min_weight') is not None:
                                constraints.append(w >= constraints_dict['min_weight'])
                            if constraints_dict.get('max_weight') is not None:
                                constraints.append(w <= constraints_dict['max_weight'])
                            
                            prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                            prob.solve(verbose=False)
                            
                            if prob.status == cp.OPTIMAL:
                                sample_vols.append(np.sqrt(prob.value) * np.sqrt(12))
                            else:
                                sample_vols.append(np.nan)
                        except:
                            sample_vols.append(np.nan)
                            
                        # Ledoit-Wolf frontier
                        try:
                            w = cp.Variable(len(final_tickers))
                            portfolio_return = mean_returns.T @ w
                            portfolio_variance = cp.quad_form(w, lw_cov)
                            
                            constraints = [cp.sum(w) == 1, portfolio_return == target_ret]
                            
                            if constraints_dict.get('min_weight') is not None:
                                constraints.append(w >= constraints_dict['min_weight'])
                            if constraints_dict.get('max_weight') is not None:
                                constraints.append(w <= constraints_dict['max_weight'])
                            
                            prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                            prob.solve(verbose=False)
                            
                            if prob.status == cp.OPTIMAL:
                                lw_vols.append(np.sqrt(prob.value) * np.sqrt(12))
                            else:
                                lw_vols.append(np.nan)
                        except:
                            lw_vols.append(np.nan)
                    
                    # Plot efficient frontiers
                    sample_vols = np.array(sample_vols)
                    lw_vols = np.array(lw_vols)
                    
                    sample_valid = ~np.isnan(sample_vols)
                    lw_valid = ~np.isnan(lw_vols)
                    
                    if np.any(sample_valid):
                        ax.plot(sample_vols[sample_valid], target_returns[sample_valid],
                               color=colors['sample'], linewidth=2, alpha=0.8, 
                               label='Sample Covariance Frontier')
                    
                    if np.any(lw_valid):
                        ax.plot(lw_vols[lw_valid], target_returns[lw_valid],
                               color=colors['ledoit_wolf'], linewidth=2, alpha=0.8, 
                               label='Ledoit-Wolf Frontier')
                    
                    # Calculate and plot minimum variance portfolios
                    # Sample minimum variance
                    w_sample = cp.Variable(len(final_tickers))
                    objective_sample = cp.Minimize(cp.quad_form(w_sample, sample_cov))
                    constraints_sample = [cp.sum(w_sample) == 1]
                    
                    if constraints_dict.get('min_weight') is not None:
                        constraints_sample.append(w_sample >= constraints_dict['min_weight'])
                    if constraints_dict.get('max_weight') is not None:
                        constraints_sample.append(w_sample <= constraints_dict['max_weight'])
                    
                    prob_sample = cp.Problem(objective_sample, constraints_sample)
                    prob_sample.solve(verbose=False)
                    
                    if prob_sample.status == cp.OPTIMAL:
                        sample_weights = w_sample.value
                        sample_ret = np.sum(sample_weights * mean_returns)
                        sample_vol = np.sqrt(objective_sample.value) * np.sqrt(12)
                        ax.scatter(sample_vol, sample_ret, color=colors['sample'], s=100, marker='o', 
                                  label='Sample Min Variance', zorder=5, edgecolors='white', linewidth=2)
                    
                    # Ledoit-Wolf minimum variance
                    w_lw = cp.Variable(len(final_tickers))
                    objective_lw = cp.Minimize(cp.quad_form(w_lw, lw_cov))
                    constraints_lw = [cp.sum(w_lw) == 1]
                    
                    if constraints_dict.get('min_weight') is not None:
                        constraints_lw.append(w_lw >= constraints_dict['min_weight'])
                    if constraints_dict.get('max_weight') is not None:
                        constraints_lw.append(w_lw <= constraints_dict['max_weight'])
                    
                    prob_lw = cp.Problem(objective_lw, constraints_lw)
                    prob_lw.solve(verbose=False)
                    
                    if prob_lw.status == cp.OPTIMAL:
                        lw_weights = w_lw.value
                        lw_ret = np.sum(lw_weights * mean_returns)
                        lw_vol = np.sqrt(objective_lw.value) * np.sqrt(12)
                        ax.scatter(lw_vol, lw_ret, color=colors['ledoit_wolf'], s=100, marker='s', 
                                  label='Ledoit-Wolf Min Variance', zorder=5, edgecolors='white', linewidth=2)
                    
                    # Add risk-free rate point
                    ax.scatter(0, risk_free_rate, color='black', s=80, marker='^', 
                              label=f'Risk-Free Rate ({risk_free_rate:.1%})', zorder=5)
                    
                    # Add capital allocation line (use sample covariance for tangency)
                    if np.any(sample_valid):
                        sharpe_ratios = (target_returns[sample_valid] - risk_free_rate) / sample_vols[sample_valid]
                        if len(sharpe_ratios) > 0 and not np.all(np.isnan(sharpe_ratios)):
                            best_sharpe_idx = np.nanargmax(sharpe_ratios)
                            tangency_ret = target_returns[sample_valid][best_sharpe_idx]
                            tangency_vol = sample_vols[sample_valid][best_sharpe_idx]
                            
                            # Draw CAL
                            max_vol = max(np.nanmax(sample_vols), np.nanmax(lw_vols)) * 1.2
                            cal_vols = np.linspace(0, max_vol, 100)
                            cal_rets = risk_free_rate + (tangency_ret - risk_free_rate) / tangency_vol * cal_vols
                            ax.plot(cal_vols, cal_rets, 'k--', alpha=0.7, linewidth=1.5, label='Capital Allocation Line')
                            
                            # Mark tangency point
                            ax.scatter(tangency_vol, tangency_ret, color='red', s=80, marker='*', 
                                      label='Tangency Portfolio', zorder=5, edgecolors='white', linewidth=1)
                    
        except Exception as e:
            ax.text(0.5, 0.5, f'Error generating efficient frontier: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.set_xlabel('Annualized Volatility', fontweight='bold')
        ax.set_ylabel('Annualized Expected Return', fontweight='bold')
        ax.set_title('Efficient Frontier Comparison (Final Estimation Period)', fontweight='bold', fontsize=16)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format axes as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        plt.tight_layout()
        return fig
        
    def create_summary_table(self, 
                            portfolio_weights: pd.DataFrame,
                            performance_metrics: Dict,
                            final_tickers: List[str],
                            figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        NEW: Create table showing weights and performance summary
        """
        
        figsize = figsize or (14, 10)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Portfolio Summary: Weights & Performance', fontsize=16, fontweight='bold')
        
        # Left side: Portfolio Weights Table
        ax1.axis('off')
        
        if not portfolio_weights.empty:
            final_date = portfolio_weights.index[-1]
            final_weights = portfolio_weights.loc[final_date]
            
            # Prepare weights data for both methods
            weights_data = []
            
            for ticker in final_tickers:
                sample_col = f'{ticker}_sample'
                lw_col = f'{ticker}_lw'
                
                sample_weight = final_weights.get(sample_col, np.nan)
                lw_weight = final_weights.get(lw_col, np.nan)
                
                if not (pd.isna(sample_weight) and pd.isna(lw_weight)):
                    weights_data.append([
                        ticker,
                        f"{sample_weight:.1%}" if not pd.isna(sample_weight) else "N/A",
                        f"{lw_weight:.1%}" if not pd.isna(lw_weight) else "N/A"
                    ])
            
            # Sort by sample weight (absolute value)
            weights_data.sort(key=lambda x: abs(float(x[1].strip('%'))/100) if x[1] != "N/A" else 0, reverse=True)
            
            if weights_data:
                weights_table = ax1.table(cellText=weights_data,
                                        colLabels=['Ticker', 'Sample Cov', 'Ledoit-Wolf'],
                                        cellLoc='center',
                                        loc='center')
                weights_table.auto_set_font_size(False)
                weights_table.set_fontsize(10)
                weights_table.scale(1, 1.5)
        
        ax1.set_title('Final Period Portfolio Weights', fontweight='bold', fontsize=14)
        
        # Right side: Performance Summary Table
        ax2.axis('off')
        
        if performance_metrics:
            perf_data = []
            metrics_to_show = [
                ('Total Return', 'total_return', '.2%'),
                ('Annualized Return', 'annualized_return', '.2%'),
                ('Annualized Volatility', 'annualized_volatility', '.2%'),
                ('Sharpe Ratio', 'sharpe_ratio', '.3f'),
                ('Sortino Ratio', 'sortino_ratio', '.3f'),
                ('Max Drawdown', 'max_drawdown', '.2%'),
                ('Win Rate', 'win_rate', '.1%'),
                ('Best Month', 'best_month', '.2%'),
                ('Worst Month', 'worst_month', '.2%'),
                ('Skewness', 'skewness', '.3f'),
                ('Kurtosis', 'kurtosis', '.3f')
            ]
            
            for metric_name, key, fmt in metrics_to_show:
                sample_val = performance_metrics.get('sample', {}).get(key, np.nan)
                lw_val = performance_metrics.get('lw', {}).get(key, np.nan)
                
                sample_str = f"{sample_val:{fmt}}" if not np.isnan(sample_val) else "N/A"
                lw_str = f"{lw_val:{fmt}}" if not np.isnan(lw_val) else "N/A"
                
                perf_data.append([metric_name, sample_str, lw_str])
            
            if perf_data:
                perf_table = ax2.table(cellText=perf_data,
                                     colLabels=['Metric', 'Sample Cov', 'Ledoit-Wolf'],
                                     cellLoc='center',
                                     loc='center')
                perf_table.auto_set_font_size(False)
                perf_table.set_fontsize(10)
                perf_table.scale(1, 1.5)
        
        ax2.set_title('Performance Summary', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        return fig
        """
        Create comprehensive performance dashboard with your requested plots
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            Backtest results
        portfolio_weights : pd.DataFrame
            Portfolio weights over time
        performance_metrics : Dict
            Performance metrics
        config : Dict
            Configuration with risk_free_rate
        returns_data : pd.DataFrame, optional
            Returns data for efficient frontier
        figsize : Tuple[int, int], optional
            Figure size override
            
        Returns:
        --------
        plt.Figure
            Complete dashboard figure
        """
        
        figsize = figsize or (20, 15)
        fig = plt.figure(figsize=figsize)
        fig.suptitle('Portfolio Optimization Performance Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # Create grid layout - 3 rows, 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        colors = self.color_schemes[self.style]
        
        # 1. Cumulative Returns (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    cumulative = (1 + returns).cumprod()
                    ax1.plot(cumulative.index, cumulative.values,
                           color=colors[color_key], linewidth=3, 
                           label=label, alpha=0.9)
                    
        ax1.set_title('Cumulative Returns', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{(y-1):.0%}'))
        
        # 2. Key Metrics Table (1 column)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        # Create metrics table
        table_data = []
        metrics_to_show = [
            ('Sharpe Ratio', 'sharpe_ratio', '.3f'),
            ('Annual Return', 'annualized_return', '.1%'),
            ('Annual Vol', 'annualized_volatility', '.1%'),
            ('Max Drawdown', 'max_drawdown', '.1%')
        ]
        
        for metric_name, key, fmt in metrics_to_show:
            sample_val = performance_metrics.get('sample', {}).get(key, 0)
            lw_val = performance_metrics.get('lw', {}).get(key, 0)
            table_data.append([metric_name, f"{sample_val:{fmt}}", f"{lw_val:{fmt}}"])
            
        table = ax2.table(cellText=table_data,
                         colLabels=['Metric', 'Sample Cov', 'Ledoit-Wolf'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax2.set_title('Key Performance Metrics', fontweight='bold', fontsize=14)
        
        # 3. Rolling Sharpe (spans 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    rolling_sharpe = returns.rolling(12).apply(
                        lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
                    )
                    ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                            color=colors[color_key], linewidth=2, label=label, alpha=0.8)
                    
        ax3.set_title('12-Month Rolling Sharpe Ratio', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Turnover Distribution (1 column) - REPLACED concentration
        ax4 = fig.add_subplot(gs[1, 2])
        turnover_data = []
        labels = []
        colors_list = []
        
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            turnover_series = self._calculate_turnover(portfolio_weights, method)
            if len(turnover_series) > 0:
                turnover_data.append(turnover_series.values)
                labels.append(label)
                colors_list.append(colors[color_key])
                
        if turnover_data:
            box_plot = ax4.boxplot(turnover_data, labels=labels, patch_artist=True)
            for patch, color in zip(box_plot['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
        ax4.set_title('Turnover Distribution', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Monthly Turnover')
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 5. Portfolio Weights Table (1 column) - NEW
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        
        # Get final period weights for both methods
        if not portfolio_weights.empty:
            final_date = portfolio_weights.index[-1]
            final_weights = portfolio_weights.loc[final_date]
            
            # Get top 5 positions for sample method
            sample_cols = [col for col in portfolio_weights.columns if col.endswith('_sample')]
            if sample_cols:
                sample_weights_data = []
                for col in sample_cols:
                    if col in final_weights and not pd.isna(final_weights[col]):
                        ticker = col.replace('_sample', '')
                        weight = final_weights[col]
                        sample_weights_data.append([ticker, f"{weight:.1%}"])
                
                # Sort by absolute weight and take top 5
                sample_weights_data.sort(key=lambda x: abs(float(x[1].strip('%'))/100), reverse=True)
                top_weights = sample_weights_data[:5]
                
                if top_weights:
                    weights_table = ax5.table(cellText=top_weights,
                                            colLabels=['Ticker', 'Weight'],
                                            cellLoc='center',
                                            loc='center')
                    weights_table.auto_set_font_size(False)
                    weights_table.set_fontsize(9)
                    weights_table.scale(1, 1.2)
                    
        ax5.set_title('Top 5 Holdings\n(Sample Method)', fontweight='bold', fontsize=12)
        
        # 6. Drawdown Analysis (1 column)
        ax6 = fig.add_subplot(gs[2, 1])
        for method, color_key, label in [
            ('sample', 'sample', 'Sample Covariance'),
            ('lw', 'ledoit_wolf', 'Ledoit-Wolf')
        ]:
            return_col = f'{method}_return'
            if return_col in backtest_results.columns:
                returns = backtest_results[return_col].dropna()
                if len(returns) > 0:
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    ax6.fill_between(drawdown.index, drawdown.values, 0,
                                   color=colors[color_key], alpha=0.6, label=label)
                    
        ax6.set_title('Drawdown Evolution', fontweight='bold', fontsize=12)
        ax6.set_ylabel('Drawdown')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # 7. Efficient Frontier with CAL (1 column) - NEW
        ax7 = fig.add_subplot(gs[2, 2])
        
        if returns_data is not None and not returns_data.empty:
            try:
                # Calculate efficient frontier for final period
                final_tickers = config.get('final_tickers', config.get('tickers', []))
                estimation_window = config.get('estimation_window', 36)
                risk_free_rate = config.get('risk_free_rate', 0.042)
                
                # Use last estimation window
                all_dates = returns_data.index.sort_values()
                if len(all_dates) >= estimation_window:
                    end_date = all_dates[-1]
                    start_idx = len(all_dates) - estimation_window
                    start_date = all_dates[start_idx]
                    
                    final_period_data = returns_data[final_tickers].loc[start_date:end_date].ffill().dropna()
                    
                    if len(final_period_data) >= 12:
                        # Calculate mean returns and covariance
                        mean_returns = final_period_data.mean().values * 12  # Annualized
                        sample_cov = final_period_data.cov().values
                        
                        # Generate efficient frontier points
                        target_returns = np.linspace(mean_returns.min(), mean_returns.max() * 1.5, 20)
                        frontier_vols = []
                        
                        for target_ret in target_returns:
                            try:
                                import cvxpy as cp
                                w = cp.Variable(len(final_tickers))
                                portfolio_return = mean_returns.T @ w
                                portfolio_variance = cp.quad_form(w, sample_cov)
                                
                                constraints = [cp.sum(w) == 1, portfolio_return == target_ret]
                                
                                # Add constraints from config
                                constraints_dict = config.get('constraints', {})
                                if constraints_dict.get('min_weight') is not None:
                                    constraints.append(w >= constraints_dict['min_weight'])
                                if constraints_dict.get('max_weight') is not None:
                                    constraints.append(w <= constraints_dict['max_weight'])
                                
                                prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                                prob.solve(verbose=False)
                                
                                if prob.status == cp.OPTIMAL:
                                    frontier_vols.append(np.sqrt(prob.value) * np.sqrt(12))
                                else:
                                    frontier_vols.append(np.nan)
                            except:
                                frontier_vols.append(np.nan)
                        
                        # Plot efficient frontier
                        frontier_vols = np.array(frontier_vols)
                        valid_mask = ~np.isnan(frontier_vols)
                        
                        if np.any(valid_mask):
                            ax7.plot(frontier_vols[valid_mask], target_returns[valid_mask],
                                   'b-', linewidth=2, alpha=0.8, label='Efficient Frontier')
                            
                            # Add capital allocation line
                            if np.any(valid_mask):
                                # Find tangency portfolio (max Sharpe ratio)
                                sharpe_ratios = (target_returns[valid_mask] - risk_free_rate) / frontier_vols[valid_mask]
                                if len(sharpe_ratios) > 0 and not np.all(np.isnan(sharpe_ratios)):
                                    best_idx = np.nanargmax(sharpe_ratios)
                                    tangency_ret = target_returns[valid_mask][best_idx]
                                    tangency_vol = frontier_vols[valid_mask][best_idx]
                                    
                                    # Draw CAL
                                    max_vol = np.nanmax(frontier_vols) * 1.2
                                    cal_vols = np.linspace(0, max_vol, 50)
                                    cal_rets = risk_free_rate + (tangency_ret - risk_free_rate) / tangency_vol * cal_vols
                                    ax7.plot(cal_vols, cal_rets, 'g--', alpha=0.7, linewidth=1.5, label='CAL')
                                    
                                    # Mark tangency point
                                    ax7.scatter(tangency_vol, tangency_ret, color='red', s=50, zorder=5, label='Tangency')
                            
                            # Mark risk-free rate
                            ax7.scatter(0, risk_free_rate, color='green', s=40, marker='^', zorder=5, label='Risk-Free')
                        
            except Exception as e:
                ax7.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                       ha='center', va='center', transform=ax7.transAxes, fontsize=10)
        else:
            ax7.text(0.5, 0.5, 'Returns data\nnot available', 
                   ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        
        ax7.set_title('Efficient Frontier & CAL', fontweight='bold', fontsize=12)
        ax7.set_xlabel('Volatility')
        ax7.set_ylabel('Expected Return')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax7.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        return fig
        
    def _calculate_turnover(self, weights_df: pd.DataFrame, method: str = 'sample') -> pd.Series:
        """Calculate portfolio turnover for plotting"""
        
        method_suffix = '_sample' if method == 'sample' else '_lw'
        weight_cols = [col for col in weights_df.columns if col.endswith(method_suffix)]
        
        if len(weight_cols) == 0:
            return pd.Series(dtype=float)
            
        weight_matrix = weights_df[weight_cols].fillna(0)
        
        turnover_list = []
        dates = weight_matrix.index
        
        for i in range(1, len(weight_matrix)):
            current_weights = weight_matrix.iloc[i].values
            previous_weights = weight_matrix.iloc[i-1].values
            turnover = np.sum(np.abs(current_weights - previous_weights))
            turnover_list.append(turnover)
            
        return pd.Series(turnover_list, index=dates[1:])
        
    def _set_plot_style(self):
        """Set plotting style based on preferences"""
        
        if self.style == 'modern':
            plt.rcParams.update({
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 16,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white'
            })
        elif self.style == 'minimal':
            plt.rcParams.update({
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.left': False,
                'axes.spines.bottom': False,
                'xtick.bottom': False,
                'xtick.top': False,
                'ytick.left': False,
                'ytick.right': False
            })
            
    def save_all_plots(self, 
                      backtest_results: pd.DataFrame,
                      portfolio_weights: pd.DataFrame,
                      performance_metrics: Dict,
                      output_dir: str = "plots",
                      dpi: int = 300) -> List[str]:
        """
        Generate and save all standard plots
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            Backtest results
        portfolio_weights : pd.DataFrame
            Portfolio weights
        performance_metrics : Dict
            Performance metrics
        output_dir : str
            Output directory for plots
        dpi : int
            Resolution for saved plots
            
        Returns:
        --------
        List[str]
            List of saved file paths
        """
        
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate and save each plot type
        plots_to_generate = [
            ('cumulative_returns', self.plot_cumulative_returns, backtest_results),
            ('rolling_metrics', self.plot_rolling_metrics, backtest_results),
            ('turnover_analysis', self.plot_turnover_analysis, portfolio_weights),
            ('risk_comparison', self.plot_risk_metrics_comparison, performance_metrics),
            ('dashboard', self.create_performance_dashboard, 
             backtest_results, portfolio_weights, performance_metrics)
        ]
        
        for plot_name, plot_func, *args in plots_to_generate:
            try:
                if plot_name == 'dashboard':
                    fig = plot_func(*args)
                else:
                    fig = plot_func(*args)
                    
                filename = f"{output_dir}/{plot_name}_{timestamp}.png"
                fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
                saved_files.append(filename)
                plt.close(fig)
                
                self.logger.info(f"Saved plot: {filename}")
                
            except Exception as e:
                self.logger.error(f"Error generating {plot_name}: {e}")
                
        return saved_files