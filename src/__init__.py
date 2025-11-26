# src/__init__.py
"""
Intelligent Portfolio Optimization
"""

__version__ = "0.1.0"
__author__ = "Shahmir Javed"
__email__ = "TBA"

# Main imports for easy access
from .portfolio.optimizer import PortfolioOptimizer
from .ai.analyzer import AIPortfolioAnalyzer

__all__ = [
    "PortfolioOptimizer", 
    "AIPortfolioAnalyzer"
]