"""
LangChain-based AI Portfolio Analyzer
Supports both OpenAI (with API key) and local models (without key)
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
logging.getLogger("langchain").setLevel(logging.ERROR)

# Check for LangChain availability
try:
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFacePipeline
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not installed. Install with: pip install langchain langchain-openai langchain-huggingface")

# Check for transformers (for local models)
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .prompts import PortfolioPrompts


class LangChainPortfolioAnalyzer:
    """
    AI Portfolio Analyzer using LangChain
    Supports OpenAI GPT models (with API key) or local Hugging Face models (without key)
    """

    def __init__(self, use_openai: bool = None, model_name: Optional[str] = None):
        """
        Initialize the LangChain analyzer

        Args:
            use_openai: If True, use OpenAI. If False, use local models. If None, auto-detect
            model_name: Specific model to use (optional)
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain required. Install with: pip install langchain openai python-dotenv")

        self.prompts = PortfolioPrompts()

        # Auto-detect OpenAI availability if not specified
        if use_openai is None:
            openai_key = os.getenv('OPENAI_API_KEY')
            use_openai = openai_key is not None and openai_key.strip() != ''

        self.use_openai = use_openai

        # Initialize the appropriate model
        if self.use_openai:
            self._initialize_openai(model_name)
        else:
            self._initialize_local(model_name)

    def _initialize_openai(self, model_name: Optional[str] = None):
        """Initialize OpenAI model"""
        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Please set OPENAI_API_KEY in .env file or use local models instead."
            )

        model_name = model_name or "gpt-4o-mini"

        print(f"Initializing OpenAI model: {model_name}")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=api_key
        )

        self.model_info = {
            'provider': 'OpenAI',
            'model': model_name,
            'type': 'cloud'
        }

        print(f"OpenAI {model_name} initialized")

    def _initialize_local(self, model_name: Optional[str] = None):
        """Initialize local Hugging Face model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for local models. Install with: pip install transformers torch")

        model_name = model_name or "mistralai/Mistral-7B-Instruct-v0.2"

        print(f"Initializing local model: {model_name}")
        print("Note: First download takes ~5-10 minutes")

        # Create Hugging Face pipeline with Mistral-7B-Instruct-v0.2
        pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=512,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        self.model_info = {
            'provider': 'Hugging Face',
            'model': model_name,
            'type': 'local'
        }

        print(f"Local model {model_name} initialized")

    def _generate_analysis(self, prompt_template: str, **kwargs) -> str:
        """Generate analysis using the configured LLM"""
        try:
            # Create prompt
            prompt = PromptTemplate(
                input_variables=list(kwargs.keys()),
                template=prompt_template
            )

            # Create chain using LCEL
            chain = prompt | self.llm | StrOutputParser()

            # Generate
            result = chain.invoke(kwargs)

            return result.strip()

        except Exception as e:
            print(f"Generation error: {e}")
            return f"Analysis unavailable: {str(e)}"

    def generate_executive_summary(self, performance_metrics: Dict, config: Dict) -> str:
        """Generate executive summary"""
        sample_sharpe = performance_metrics['sample'].get('sharpe_ratio', 0)
        lw_sharpe = performance_metrics['lw'].get('sharpe_ratio', 0)
        better_method = "Ledoit-Wolf" if lw_sharpe > sample_sharpe else "Sample Covariance"

        prompt_template = """You are a financial analyst reviewing portfolio performance.

Portfolio Configuration:
- Period: {start_year} to {end_year}
- Assets: {num_assets} stocks
- Strategy: Minimum variance optimization

Performance Results:
- Sample Covariance Sharpe: {sample_sharpe:.3f}
- Ledoit-Wolf Sharpe: {lw_sharpe:.3f}
- Better Method: {better_method}

Write a brief executive summary (2-3 sentences) highlighting the key findings."""

        return self._generate_analysis(
            prompt_template,
            start_year=config.get('start_year', 'N/A'),
            end_year=config.get('end_year', 'N/A'),
            num_assets=len(config.get('final_tickers', [])),
            sample_sharpe=sample_sharpe,
            lw_sharpe=lw_sharpe,
            better_method=better_method
        )

    def generate_investment_recommendations(self, performance_metrics: Dict, config: Dict) -> str:
        """Generate investment recommendations"""
        sample_metrics = performance_metrics['sample']
        lw_metrics = performance_metrics['lw']

        prompt_template = """You are a portfolio advisor providing investment recommendations.

Performance Analysis:
- Sample Covariance: Return {sample_return:.2%}, Volatility {sample_vol:.2%}, Sharpe {sample_sharpe:.3f}
- Ledoit-Wolf: Return {lw_return:.2%}, Volatility {lw_vol:.2%}, Sharpe {lw_sharpe:.3f}

Portfolio Constraints:
- Long-only: {long_only}
- Max weight per stock: {max_weight:.0%}

Provide 3-4 actionable investment recommendations based on these results."""

        return self._generate_analysis(
            prompt_template,
            sample_return=sample_metrics.get('annualized_return', 0),
            sample_vol=sample_metrics.get('annualized_volatility', 0),
            sample_sharpe=sample_metrics.get('sharpe_ratio', 0),
            lw_return=lw_metrics.get('annualized_return', 0),
            lw_vol=lw_metrics.get('annualized_volatility', 0),
            lw_sharpe=lw_metrics.get('sharpe_ratio', 0),
            long_only=config.get('constraints', {}).get('long_only', True),
            max_weight=config.get('constraints', {}).get('max_weight', 1.0)
        )

    def generate_risk_analysis(self, performance_metrics: Dict) -> str:
        """Generate risk analysis"""
        sample_metrics = performance_metrics['sample']

        prompt_template = """You are a risk analyst reviewing portfolio risk metrics.

Risk Metrics:
- Maximum Drawdown: {max_dd:.2%}
- Volatility: {volatility:.2%}
- Downside Risk (implied from Sortino): Consider

Provide a brief risk assessment (2-3 sentences) focusing on downside protection and volatility."""

        return self._generate_analysis(
            prompt_template,
            max_dd=sample_metrics.get('max_drawdown', 0),
            volatility=sample_metrics.get('annualized_volatility', 0)
        )

    def generate_complete_report(self, performance_metrics: Dict, portfolio_weights: Dict,
                                 turnover_metrics: Dict, config: Dict) -> Dict[str, str]:
        """Generate complete portfolio analysis report"""
        print(f"Generating analysis using {self.model_info['provider']} ({self.model_info['model']})...")

        report = {
            'generated_at': datetime.now().isoformat(),
            'model_provider': self.model_info['provider'],
            'model_name': self.model_info['model'],
            'model_type': self.model_info['type']
        }

        try:
            # Generate each section
            print("  Generating executive summary...")
            report['executive_summary'] = self.generate_executive_summary(performance_metrics, config)

            print("  Generating investment recommendations...")
            report['investment_recommendations'] = self.generate_investment_recommendations(
                performance_metrics, config
            )

            print("  Generating risk analysis...")
            report['risk_analysis'] = self.generate_risk_analysis(performance_metrics)

            # Add methodology note
            report['methodology_note'] = self._get_methodology_note(performance_metrics, config)

            print("AI analysis completed")

        except Exception as e:
            print(f"Error during generation: {e}")
            report['error'] = str(e)

        return report

    def _get_methodology_note(self, performance_metrics: Dict, config: Dict) -> str:
        """Generate methodology explanation"""
        return f"""
This analysis compared two covariance estimation methods:
1. Sample Covariance: Traditional empirical covariance matrix
2. Ledoit-Wolf Shrinkage: Regularized covariance matrix to reduce estimation error

Both methods were used to construct minimum variance portfolios over a {config.get('estimation_window', 'N/A')}-month
rolling window from {config.get('start_year', 'N/A')} to {config.get('end_year', 'N/A')}.

The Ledoit-Wolf method typically performs better when the number of assets is large relative to the
number of observations, as it reduces estimation noise in the covariance matrix.
"""
