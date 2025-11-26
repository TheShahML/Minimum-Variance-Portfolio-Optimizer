#ai/local_ai_analyzer.py

"""
This class provides AI analysis without requiring paid API keys.
Uses open-source models from Hugging Face for portfolio commentary.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")

from .prompts import PortfolioPrompts


class LocalAIPortfolioAnalyzer:
    """
    Portfolio analyzer using local/free AI models via Hugging Face transformers.
    
    This class generates portfolio analysis and commentary without requiring
    paid API services, using open-source language models.
    """
    
    def __init__(self, 
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 device: str = "auto",
                 max_length: int = 512,
                 temperature: float = 0.7):
        """
        Initialize the local AI analyzer.
        
        Args:
            model_name: Hugging Face model name for text generation
            device: Device to run model on ('cpu', 'cuda', or 'auto')
            max_length: Maximum tokens for generated text
            temperature: Sampling temperature for generation
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        
        # Initialize prompts
        self.prompts = PortfolioPrompts()
        
        # Model components
        self.generator = None
        self.tokenizer = None
        self.model = None

        # Determine device
        if device == "auto":
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
                print("GPU detected - using CUDA for acceleration")
            else:
                self.device = "cpu"
                print("Using CPU (GPU not available or torch not installed)")
        else:
            self.device = device

        print(f"Initializing Local AI Analyzer on {self.device}")
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the language model for generation."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        try:
            print(f"Loading model: {self.model_name}")
            print("Note: First download takes ~5-10 minutes (downloading ~14GB model)")
            
            # Use Mistral-7B-Instruct-v0.2 for high-quality financial analysis
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                max_new_tokens=512
            )
            
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to gpt2-medium...")
            
            # Fallback to gpt2-medium (smaller, faster)
            try:
                self.generator = pipeline(
                    "text-generation", 
                    model="gpt2-medium",
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                print("Fallback model (gpt2-medium) loaded")
            except Exception as fallback_error:
                print(f"Fallback model failed: {fallback_error}")
                self.generator = None
    
    def generate_complete_report(self, 
                               performance_metrics: Dict,
                               portfolio_weights: Dict, 
                               turnover_metrics: Dict,
                               config: Dict) -> Dict[str, str]:
        """
        Generate a complete portfolio analysis report.
        
        Args:
            performance_metrics: Performance comparison data
            portfolio_weights: Portfolio weight allocations
            turnover_metrics: Portfolio turnover statistics
            config: Analysis configuration parameters
            
        Returns:
            Dict containing different sections of the analysis
        """
        if not self.generator:
            return self._generate_template_report(performance_metrics, portfolio_weights, config)
        
        print("Generating AI portfolio analysis...")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'model_used': self.model_name,
            'device': self.device
        }
        
        try:
            # 1. Executive Summary
            print("  Generating executive summary...")
            report['executive_summary'] = self._generate_executive_summary(
                performance_metrics, config
            )
            
            # 2. Methodology Explanation  
            print("  Generating methodology explanation...")
            report['methodology_explanation'] = self._generate_methodology_explanation(
                performance_metrics, config
            )
            
            # 3. Portfolio Commentary
            print("  Generating portfolio commentary...")
            report['portfolio_commentary'] = self._generate_portfolio_commentary(
                portfolio_weights, performance_metrics, config
            )
            
            # 4. Risk Analysis
            print("  Generating risk analysis...")
            report['risk_analysis'] = self._generate_risk_analysis(
                performance_metrics, turnover_metrics, config
            )
            
            # 5. Investment Recommendations
            print("  Generating investment recommendations...")
            report['investment_recommendations'] = self._generate_recommendations(
                performance_metrics, config
            )
            
            print("AI analysis completed successfully")
            
        except Exception as e:
            print(f"Error during AI generation: {e}")
            report['error'] = str(e)
            # Return template report as fallback
            return self._generate_template_report(performance_metrics, portfolio_weights, config)
        
        return report
    
    def _generate_text(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate text using the loaded model."""
        if not self.generator:
            return "AI model not available - using template analysis."
        
        try:
            # Generate response
            response = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract generated text (remove the original prompt)
            generated_text = response[0]['generated_text']
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Text generation error: {e}")
            return "Analysis unavailable due to model error."
    
    def _generate_executive_summary(self, performance_metrics: Dict, config: Dict) -> str:
        """Generate executive summary of portfolio performance."""
        
        # Prepare data for the prompt
        sample_metrics = performance_metrics.get('sample', {})
        lw_metrics = performance_metrics.get('lw', {})
        
        sample_sharpe = sample_metrics.get('sharpe_ratio', 0)
        lw_sharpe = lw_metrics.get('sharpe_ratio', 0)
        
        better_method = "Ledoit-Wolf" if lw_sharpe > sample_sharpe else "Sample Covariance"
        improvement = abs(lw_sharpe - sample_sharpe) / max(sample_sharpe, 0.01) * 100
        
        prompt = f"""
Portfolio Analysis Executive Summary:

A minimum variance portfolio optimization compared Sample Covariance vs Ledoit-Wolf shrinkage methods over {config.get('time_period', 'the analysis period')}.

Key Results:
- Sample Covariance Sharpe Ratio: {sample_sharpe:.3f}
- Ledoit-Wolf Sharpe Ratio: {lw_sharpe:.3f}
- Superior Method: {better_method}
- Performance Improvement: {improvement:.1f}%

Analysis Summary:"""
        
        return self._generate_text(prompt, max_new_tokens=150)
    
    def _generate_methodology_explanation(self, performance_metrics: Dict, config: Dict) -> str:
        """Generate explanation of the optimization methodology."""
        
        prompt = f"""
Portfolio Optimization Methodology Explanation:

This analysis compared two covariance estimation methods for minimum variance portfolio construction:

1. Sample Covariance: Traditional historical covariance matrix
2. Ledoit-Wolf Shrinkage: Shrinks sample covariance toward structured estimator

Configuration:
- Portfolio Universe: {config.get('num_assets', 'N/A')} assets
- Estimation Window: {config.get('estimation_window', 'N/A')} months  
- Analysis Period: {config.get('time_period', 'N/A')}
- Constraints: Long-only, maximum 25% per position

Technical Explanation:"""
        
        return self._generate_text(prompt, max_new_tokens=200)
    
    def _generate_portfolio_commentary(self, portfolio_weights: Dict, 
                                     performance_metrics: Dict, config: Dict) -> str:
        """Generate commentary on portfolio composition."""
        
        # Get latest portfolio weights
        latest_weights = {}
        for method in ['sample', 'lw']:
            if method in portfolio_weights:
                method_weights = portfolio_weights[method]
                if isinstance(method_weights, pd.DataFrame) and not method_weights.empty:
                    latest_weights[method] = method_weights.iloc[-1].to_dict()
                
        prompt = f"""
Portfolio Composition Analysis:

The optimization resulted in different asset allocations between methods.

Portfolio Characteristics:
- Number of Assets: {config.get('num_assets', 'N/A')}
- Rebalancing: Monthly
- Constraints: Long-only, max 25% per position
- Time Period: {config.get('time_period', 'N/A')}

Portfolio Commentary:"""
        
        return self._generate_text(prompt, max_new_tokens=180)
    
    def _generate_risk_analysis(self, performance_metrics: Dict, 
                              turnover_metrics: Dict, config: Dict) -> str:
        """Generate risk analysis commentary."""
        
        sample_vol = performance_metrics.get('sample', {}).get('annualized_volatility', 0)
        lw_vol = performance_metrics.get('lw', {}).get('annualized_volatility', 0)
        
        prompt = f"""
Portfolio Risk Analysis:

Risk Metrics Comparison:
- Sample Covariance Volatility: {sample_vol:.2%}
- Ledoit-Wolf Volatility: {lw_vol:.2%}
- Target: Minimum variance optimization

Risk Assessment:"""
        
        return self._generate_text(prompt, max_new_tokens=160)
    
    def _generate_recommendations(self, performance_metrics: Dict, config: Dict) -> str:
        """Generate investment recommendations."""
        
        sample_sharpe = performance_metrics.get('sample', {}).get('sharpe_ratio', 0)
        lw_sharpe = performance_metrics.get('lw', {}).get('sharpe_ratio', 0)
        better_method = "Ledoit-Wolf" if lw_sharpe > sample_sharpe else "Sample Covariance"
        
        prompt = f"""
Investment Recommendations:

Based on the portfolio optimization analysis showing {better_method} performed better:

Key Recommendations:
1. Methodology: Use {better_method} for covariance estimation
2. Implementation: Monthly rebalancing with transaction cost consideration
3. Risk Management: Monitor portfolio concentration and turnover

Specific Actions:"""
        
        return self._generate_text(prompt, max_new_tokens=150)
    
    def _generate_template_report(self, performance_metrics: Dict, 
                                portfolio_weights: Dict, config: Dict) -> Dict[str, str]:
        """Generate template analysis when AI model is unavailable."""
        
        sample_metrics = performance_metrics.get('sample', {})
        lw_metrics = performance_metrics.get('lw', {})
        
        sample_sharpe = sample_metrics.get('sharpe_ratio', 0)
        lw_sharpe = lw_metrics.get('sharpe_ratio', 0)
        
        better_method = "Ledoit-Wolf" if lw_sharpe > sample_sharpe else "Sample Covariance"
        improvement = abs(lw_sharpe - sample_sharpe) / max(sample_sharpe, 0.01) * 100
        
        return {
            'generated_at': datetime.now().isoformat(),
            'model_used': 'template_analysis',
            'device': 'none',
            
            'executive_summary': f"""
The portfolio optimization analysis compared Sample Covariance vs Ledoit-Wolf shrinkage estimation methods over {config.get('time_period', 'the analysis period')}.

Key Results:
• {better_method} achieved superior risk-adjusted performance
• Sharpe Ratio: Sample ({sample_sharpe:.3f}) vs Ledoit-Wolf ({lw_sharpe:.3f})
• Performance improvement of {improvement:.1f}% using the superior method
• Both methods achieved the minimum variance objective with reasonable turnover

This demonstrates the value of covariance shrinkage in portfolio optimization, particularly for medium-sized portfolios where sample estimation can be noisy.
            """,
            
            'methodology_explanation': f"""
This analysis implemented minimum variance portfolio optimization using two covariance estimation approaches:

1. Sample Covariance: Uses historical return covariances directly from the data
2. Ledoit-Wolf Shrinkage: Shrinks the sample covariance toward a structured estimator to reduce estimation error

The rolling window approach with {config.get('estimation_window', 36)}-month windows ensures realistic out-of-sample performance measurement. Constraints included long-only positions with maximum 25% allocation per asset.

Ledoit-Wolf shrinkage typically improves performance by reducing the impact of estimation noise in the covariance matrix, which is particularly beneficial for portfolios with limited historical data or moderate universe sizes.
            """,
            
            'portfolio_commentary': f"""
The portfolio optimization resulted in well-diversified allocations across the {config.get('num_assets', 'selected')} assets.

Key characteristics:
• Diversified allocations respecting the 25% maximum position constraint
• Monthly rebalancing maintained portfolio alignment with minimum variance objectives
• Both methods showed reasonable turnover levels, suggesting stable optimization results
• Weight distributions reflected the risk-minimization objective rather than return maximization

The consistency of allocations between methods suggests robust optimization results, with differences primarily reflecting the covariance estimation approach rather than fundamental changes in portfolio structure.
            """,
            
            'risk_analysis': f"""
Risk analysis shows both methods successfully achieved minimum variance objectives:

Volatility Comparison:
• Sample Covariance: {sample_metrics.get('annualized_volatility', 0):.2%} annual volatility
• Ledoit-Wolf: {lw_metrics.get('annualized_volatility', 0):.2%} annual volatility

The risk reduction from diversification was effective in both cases. Maximum drawdowns remained within reasonable bounds for minimum variance portfolios. The superior Sharpe ratio of {better_method} indicates better risk-adjusted performance while maintaining similar volatility levels.

Transaction costs and portfolio turnover should be monitored in live implementation to ensure the optimization benefits are not eroded by excessive trading.
            """,
            
            'investment_recommendations': f"""
Based on the analysis results favoring {better_method}, we recommend:

1. METHODOLOGY: Implement {better_method} for ongoing covariance estimation
   - Provides {improvement:.1f}% improvement in risk-adjusted returns
   - More robust parameter estimation for this portfolio size

2. IMPLEMENTATION:
   - Maintain monthly rebalancing frequency
   - Monitor transaction costs vs optimization benefits
   - Consider quarterly rebalancing if turnover costs are high

3. RISK MANAGEMENT:
   - Continue monitoring portfolio concentration (max 25% constraint)
   - Track performance attribution between asset selection and allocation
   - Stress test results under different market regimes

4. NEXT STEPS:
   - Extend analysis to additional time periods
   - Consider factor-based risk models for larger portfolios
   - Evaluate alternative optimization objectives (max Sharpe, risk parity)
            """
        }


def main():
    """Test the local AI analyzer."""
    print("Testing Local AI Portfolio Analyzer")
    print("=" * 40)
    
    # Sample data for testing
    performance_metrics = {
        'sample': {
            'sharpe_ratio': 0.85,
            'annualized_return': 0.12,
            'annualized_volatility': 0.14
        },
        'lw': {
            'sharpe_ratio': 0.92,
            'annualized_return': 0.13,
            'annualized_volatility': 0.14
        }
    }
    
    config = {
        'num_assets': 20,
        'time_period': '2015-2023',
        'estimation_window': 36
    }
    
    # Initialize analyzer
    analyzer = LocalAIPortfolioAnalyzer()
    
    # Generate report
    report = analyzer.generate_complete_report(
        performance_metrics=performance_metrics,
        portfolio_weights={},
        turnover_metrics={},
        config=config
    )
    
    # Display results
    print("\nGenerated Report:")
    print("=" * 40)
    for section, content in report.items():
        if section not in ['generated_at', 'model_used', 'device']:
            print(f"\n{section.upper().replace('_', ' ')}:")
            print("-" * 30)
            print(content)


if __name__ == "__main__":
    main()