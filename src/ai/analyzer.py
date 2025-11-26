# src/ai/analyzer.py
"""
AI-powered portfolio analysis using LangChain and OpenAI
Generates natural language insights from optimization results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json

try:
    from langchain_openai import ChatOpenAI, OpenAI
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    
from .prompts import PortfolioPrompts

class AIPortfolioAnalyzer:
    """
    AI-powered portfolio analysis using LangChain
    Generates insights, explanations, and recommendations from optimization results
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.3,
                 max_tokens: int = 1000,
                 api_key: Optional[str] = None):
        """
        Initialize AI Portfolio Analyzer
        
        Parameters:
        -----------
        model_name : str
            OpenAI model to use (default: gpt-3.5-turbo)
        temperature : float
            Model temperature for creativity vs consistency (default: 0.3)
        max_tokens : int
            Maximum tokens in response (default: 1000)
        api_key : str, optional
            OpenAI API key. If None, expects OPENAI_API_KEY env variable
        """
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed. Run: pip install langchain openai")
            
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        try:
            if "gpt-3.5" in model_name or "gpt-4" in model_name:
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
            else:
                self.llm = OpenAI(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
            
        # Initialize prompts
        self.prompts = PortfolioPrompts()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _format_performance_data(self, performance_metrics: Dict) -> Dict:
        """Format performance metrics for LLM consumption"""
        
        formatted_data = {}
        
        for method in ['sample', 'lw']:
            if method in performance_metrics:
                metrics = performance_metrics[method]
                method_name = 'Sample Covariance' if method == 'sample' else 'Ledoit-Wolf'
                
                formatted_data[method_name] = {
                    'total_return': f"{metrics.get('total_return', 0):.2%}",
                    'annualized_return': f"{metrics.get('annualized_return', 0):.2%}",
                    'annualized_volatility': f"{metrics.get('annualized_volatility', 0):.2%}",
                    'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                    'max_drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                    'win_rate': f"{metrics.get('win_rate', 0):.1%}",
                    'best_month': f"{metrics.get('best_month', 0):.2%}",
                    'worst_month': f"{metrics.get('worst_month', 0):.2%}"
                }
                
        return formatted_data
        
    def _format_portfolio_weights(self, portfolio_weights: pd.DataFrame, 
                                final_tickers: List[str], top_n: int = 10) -> Dict:
        """Format portfolio weights for LLM consumption"""
        
        if portfolio_weights.empty:
            return {}
            
        # Get final period weights
        final_date = portfolio_weights.index[-1]
        final_weights = portfolio_weights.loc[final_date]
        
        result = {}
        
        for method in ['sample', 'lw']:
            method_name = 'Sample Covariance' if method == 'sample' else 'Ledoit-Wolf'
            method_suffix = '_sample' if method == 'sample' else '_lw'
            
            # Extract weights for this method
            weights_data = []
            for ticker in final_tickers:
                col_name = f'{ticker}{method_suffix}'
                if col_name in final_weights and not pd.isna(final_weights[col_name]):
                    weight = final_weights[col_name]
                    weights_data.append({
                        'ticker': ticker,
                        'weight': weight,
                        'abs_weight': abs(weight)
                    })
                    
            # Sort by absolute weight and take top N
            weights_data.sort(key=lambda x: x['abs_weight'], reverse=True)
            top_weights = weights_data[:top_n]
            
            # Format for LLM
            formatted_weights = []
            for item in top_weights:
                formatted_weights.append({
                    'ticker': item['ticker'],
                    'weight': f"{item['weight']:.2%}",
                    'position': 'Long' if item['weight'] > 0 else 'Short' if item['weight'] < 0 else 'Zero'
                })
                
            result[method_name] = formatted_weights
            
        return result
        
    def generate_performance_summary(self, 
                                   performance_metrics: Dict,
                                   config: Dict) -> str:
        """
        Generate AI summary of portfolio performance
        
        Parameters:
        -----------
        performance_metrics : Dict
            Performance metrics from PerformanceAnalyzer
        config : Dict
            Portfolio configuration parameters
            
        Returns:
        --------
        str
            Natural language performance summary
        """
        
        try:
            # Format data for LLM
            formatted_metrics = self._format_performance_data(performance_metrics)
            
            # Create context
            context = {
                'performance_data': formatted_metrics,
                'num_assets': len(config.get('final_tickers', config.get('tickers', []))),
                'time_period': f"{config.get('start_year')}-{config.get('end_year')}",
                'estimation_window': f"{config.get('estimation_window')} months",
                'risk_free_rate': f"{config.get('risk_free_rate', 0.042):.1%}",
                'constraints': config.get('constraints', {}),
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Generate prompt
            prompt = self.prompts.get_performance_summary_prompt()
            
            # Create chain
            if isinstance(self.llm, ChatOpenAI):
                messages = [
                    SystemMessage(content="You are a quantitative finance expert providing portfolio analysis."),
                    HumanMessage(content=prompt.format(**context))
                ]
                response = self.llm.invoke(messages)
                return response.content
            else:
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke(context)
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return f"Error generating AI summary: {str(e)}"
            
    def generate_methodology_explanation(self, 
                                       performance_metrics: Dict,
                                       config: Dict) -> str:
        """
        Generate explanation of optimization methodology and results
        
        Parameters:
        -----------
        performance_metrics : Dict
            Performance metrics from analysis
        config : Dict
            Portfolio configuration
            
        Returns:
        --------
        str
            Explanation of methodology and why results occurred
        """
        
        try:
            # Determine which method performed better
            sample_sharpe = performance_metrics.get('sample', {}).get('sharpe_ratio', 0)
            lw_sharpe = performance_metrics.get('lw', {}).get('sharpe_ratio', 0)
            
            better_method = 'Ledoit-Wolf' if lw_sharpe > sample_sharpe else 'Sample Covariance'
            sharpe_difference = abs(lw_sharpe - sample_sharpe)
            
            # Get shrinkage information if available
            shrinkage_info = ""
            if 'lw' in performance_metrics:
                # This would need to come from backtest results
                shrinkage_info = "Shrinkage parameters adapted over time based on market conditions."
                
            context = {
                'better_method': better_method,
                'sharpe_difference': f"{sharpe_difference:.3f}",
                'sample_sharpe': f"{sample_sharpe:.3f}",
                'lw_sharpe': f"{lw_sharpe:.3f}",
                'estimation_window': config.get('estimation_window'),
                'num_assets': len(config.get('final_tickers', [])),
                'time_period': f"{config.get('start_year')}-{config.get('end_year')}",
                'shrinkage_info': shrinkage_info,
                'constraints_info': self._format_constraints_info(config.get('constraints', {}))
            }
            
            prompt = self.prompts.get_methodology_explanation_prompt()
            
            if isinstance(self.llm, ChatOpenAI):
                messages = [
                    SystemMessage(content="You are a quantitative finance professor explaining portfolio theory."),
                    HumanMessage(content=prompt.format(**context))
                ]
                response = self.llm.invoke(messages)
                return response.content
            else:
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke(context)
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating methodology explanation: {e}")
            return f"Error generating methodology explanation: {str(e)}"
            
    def generate_portfolio_commentary(self,
                                    portfolio_weights: pd.DataFrame,
                                    performance_metrics: Dict,
                                    config: Dict) -> str:
        """
        Generate commentary on portfolio composition and positioning
        
        Parameters:
        -----------
        portfolio_weights : pd.DataFrame
            Portfolio weights over time
        performance_metrics : Dict
            Performance metrics
        config : Dict
            Configuration parameters
            
        Returns:
        --------
        str
            Commentary on portfolio composition
        """
        
        try:
            # Format portfolio weights
            formatted_weights = self._format_portfolio_weights(
                portfolio_weights, 
                config.get('final_tickers', [])
            )
            
            # Calculate concentration metrics
            concentration_metrics = self._calculate_concentration_metrics(
                portfolio_weights,
                config.get('final_tickers', [])
            )
            
            context = {
                'portfolio_weights': formatted_weights,
                'concentration_metrics': concentration_metrics,
                'num_assets': len(config.get('final_tickers', [])),
                'performance_comparison': self._get_performance_comparison(performance_metrics),
                'time_period': f"{config.get('start_year')}-{config.get('end_year')}",
                'rebalance_frequency': 'Monthly',
                'constraints': config.get('constraints', {})
            }
            
            prompt = self.prompts.get_portfolio_commentary_prompt()
            
            if isinstance(self.llm, ChatOpenAI):
                messages = [
                    SystemMessage(content="You are a portfolio manager providing insights on portfolio composition."),
                    HumanMessage(content=prompt.format(**context))
                ]
                response = self.llm.invoke(messages)
                return response.content
            else:
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke(context)
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating portfolio commentary: {e}")
            return f"Error generating portfolio commentary: {str(e)}"
            
    def generate_risk_analysis(self,
                             performance_metrics: Dict,
                             turnover_metrics: Dict,
                             config: Dict) -> str:
        """
        Generate AI analysis of portfolio risk characteristics
        
        Parameters:
        -----------
        performance_metrics : Dict
            Performance metrics
        turnover_metrics : Dict
            Turnover analysis results
        config : Dict
            Configuration parameters
            
        Returns:
        --------
        str
            Risk analysis commentary
        """
        
        try:
            # Format risk metrics
            risk_data = {}
            for method in ['sample', 'lw']:
                if method in performance_metrics:
                    metrics = performance_metrics[method]
                    method_name = 'Sample Covariance' if method == 'sample' else 'Ledoit-Wolf'
                    
                    risk_data[method_name] = {
                        'volatility': f"{metrics.get('annualized_volatility', 0):.2%}",
                        'max_drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                        'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                        'sortino_ratio': f"{metrics.get('sortino_ratio', 0):.3f}",
                        'skewness': f"{metrics.get('skewness', 0):.3f}",
                        'kurtosis': f"{metrics.get('kurtosis', 0):.3f}",
                        'worst_month': f"{metrics.get('worst_month', 0):.2%}"
                    }
                    
            # Format turnover data
            turnover_data = {}
            for method in ['sample', 'lw']:
                if method in turnover_metrics:
                    method_name = 'Sample Covariance' if method == 'sample' else 'Ledoit-Wolf'
                    turnover_data[method_name] = {
                        'mean_turnover': f"{turnover_metrics[method].get('mean_turnover', 0):.1%}",
                        'max_turnover': f"{turnover_metrics[method].get('max_turnover', 0):.1%}"
                    }
                    
            context = {
                'risk_metrics': risk_data,
                'turnover_metrics': turnover_data,
                'estimation_window': f"{config.get('estimation_window')} months",
                'time_period': f"{config.get('start_year')}-{config.get('end_year')}",
                'risk_free_rate': f"{config.get('risk_free_rate', 0.042):.1%}",
                'constraints': config.get('constraints', {})
            }
            
            prompt = self.prompts.get_risk_analysis_prompt()
            
            if isinstance(self.llm, ChatOpenAI):
                messages = [
                    SystemMessage(content="You are a risk management expert analyzing portfolio characteristics."),
                    HumanMessage(content=prompt.format(**context))
                ]
                response = self.llm.invoke(messages)
                return response.content
            else:
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke(context)
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating risk analysis: {e}")
            return f"Error generating risk analysis: {str(e)}"
            
    def generate_investment_recommendations(self,
                                          performance_metrics: Dict,
                                          config: Dict) -> str:
        """
        Generate actionable investment recommendations based on results
        
        Parameters:
        -----------
        performance_metrics : Dict
            Performance analysis results
        config : Dict
            Portfolio configuration
            
        Returns:
        --------
        str
            Investment recommendations
        """
        
        try:
            # Analyze which method performed better and why
            sample_metrics = performance_metrics.get('sample', {})
            lw_metrics = performance_metrics.get('lw', {})
            
            recommendations_context = {
                'better_method': 'Ledoit-Wolf' if lw_metrics.get('sharpe_ratio', 0) > sample_metrics.get('sharpe_ratio', 0) else 'Sample Covariance',
                'sample_sharpe': f"{sample_metrics.get('sharpe_ratio', 0):.3f}",
                'lw_sharpe': f"{lw_metrics.get('sharpe_ratio', 0):.3f}",
                'sample_volatility': f"{sample_metrics.get('annualized_volatility', 0):.2%}",
                'lw_volatility': f"{lw_metrics.get('annualized_volatility', 0):.2%}",
                'time_period': f"{config.get('start_year')}-{config.get('end_year')}",
                'num_assets': len(config.get('final_tickers', [])),
                'estimation_window': config.get('estimation_window'),
                'current_constraints': config.get('constraints', {}),
                'analysis_type': 'Minimum Variance Portfolio Optimization'
            }
            
            prompt = self.prompts.get_recommendations_prompt()
            
            if isinstance(self.llm, ChatOpenAI):
                messages = [
                    SystemMessage(content="You are an investment advisor providing actionable portfolio recommendations."),
                    HumanMessage(content=prompt.format(**recommendations_context))
                ]
                response = self.llm.invoke(messages)
                return response.content
            else:
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke(recommendations_context)
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return f"Error generating recommendations: {str(e)}"
            
    def generate_complete_report(self,
                               performance_metrics: Dict,
                               portfolio_weights: pd.DataFrame,
                               turnover_metrics: Dict,
                               config: Dict) -> Dict[str, str]:
        """
        Generate complete AI analysis report
        
        Parameters:
        -----------
        performance_metrics : Dict
            Performance analysis results
        portfolio_weights : pd.DataFrame
            Portfolio weights over time
        turnover_metrics : Dict
            Turnover analysis results
        config : Dict
            Portfolio configuration
            
        Returns:
        --------
        Dict[str, str]
            Dictionary with different sections of AI analysis
        """
        
        self.logger.info("Generating complete AI analysis report...")
        
        report_sections = {}
        
        try:
            # Generate each section
            report_sections['executive_summary'] = self.generate_performance_summary(
                performance_metrics, config
            )
            
            report_sections['methodology_explanation'] = self.generate_methodology_explanation(
                performance_metrics, config
            )
            
            report_sections['portfolio_commentary'] = self.generate_portfolio_commentary(
                portfolio_weights, performance_metrics, config
            )
            
            report_sections['risk_analysis'] = self.generate_risk_analysis(
                performance_metrics, turnover_metrics, config
            )
            
            report_sections['recommendations'] = self.generate_investment_recommendations(
                performance_metrics, config
            )
            
            # Generate timestamp
            report_sections['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self.logger.info("AI analysis report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating complete report: {e}")
            report_sections['error'] = f"Error generating AI report: {str(e)}"
            
        return report_sections
        
    def _format_constraints_info(self, constraints: Dict) -> str:
        """Format constraints information for LLM"""
        
        info_parts = []
        
        if constraints.get('long_only', False):
            info_parts.append("long-only (no short selling)")
        else:
            info_parts.append("long/short allowed")
            
        if 'max_weight' in constraints and constraints['max_weight'] < 1.0:
            info_parts.append(f"maximum position size: {constraints['max_weight']:.1%}")
            
        if 'min_weight' in constraints and constraints['min_weight'] > 0:
            info_parts.append(f"minimum position size: {constraints['min_weight']:.1%}")
            
        return ", ".join(info_parts) if info_parts else "unconstrained"
        
    def _calculate_concentration_metrics(self, 
                                       portfolio_weights: pd.DataFrame,
                                       final_tickers: List[str]) -> Dict:
        """Calculate portfolio concentration metrics"""
        
        if portfolio_weights.empty:
            return {}
            
        try:
            # Get final period weights
            final_date = portfolio_weights.index[-1]
            final_weights = portfolio_weights.loc[final_date]
            
            metrics = {}
            
            for method in ['sample', 'lw']:
                method_suffix = '_sample' if method == 'sample' else '_lw'
                method_name = 'Sample Covariance' if method == 'sample' else 'Ledoit-Wolf'
                
                # Get weights for this method
                weights = []
                for ticker in final_tickers:
                    col_name = f'{ticker}{method_suffix}'
                    if col_name in final_weights and not pd.isna(final_weights[col_name]):
                        weights.append(abs(final_weights[col_name]))
                        
                if weights:
                    weights = np.array(weights)
                    # Calculate Herfindahl-Hirschman Index (concentration)
                    hhi = np.sum(weights ** 2)
                    # Top 3 concentration
                    top_3_concentration = np.sum(np.sort(weights)[-3:])
                    # Effective number of positions
                    effective_positions = 1 / hhi if hhi > 0 else 0
                    
                    metrics[method_name] = {
                        'hhi': f"{hhi:.3f}",
                        'top_3_concentration': f"{top_3_concentration:.1%}",
                        'effective_positions': f"{effective_positions:.1f}",
                        'max_position': f"{np.max(weights):.1%}" if len(weights) > 0 else "0.0%"
                    }
                    
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration metrics: {e}")
            return {}
            
    def _get_performance_comparison(self, performance_metrics: Dict) -> str:
        """Get simple performance comparison between methods"""
        
        sample_sharpe = performance_metrics.get('sample', {}).get('sharpe_ratio', 0)
        lw_sharpe = performance_metrics.get('lw', {}).get('sharpe_ratio', 0)
        
        if lw_sharpe > sample_sharpe:
            diff = ((lw_sharpe - sample_sharpe) / sample_sharpe * 100) if sample_sharpe != 0 else 0
            return f"Ledoit-Wolf outperformed by {diff:.1f}% in risk-adjusted returns"
        elif sample_sharpe > lw_sharpe:
            diff = ((sample_sharpe - lw_sharpe) / lw_sharpe * 100) if lw_sharpe != 0 else 0
            return f"Sample Covariance outperformed by {diff:.1f}% in risk-adjusted returns"
        else:
            return "Both methods showed similar risk-adjusted performance"
            
    def chat_with_results(self, 
                         user_question: str,
                         performance_metrics: Dict,
                         config: Dict,
                         context_data: Optional[Dict] = None) -> str:
        """
        Interactive chat about portfolio results
        
        Parameters:
        -----------
        user_question : str
            User's question about the portfolio
        performance_metrics : Dict
            Performance analysis results
        config : Dict
            Portfolio configuration
        context_data : Dict, optional
            Additional context data (weights, turnover, etc.)
            
        Returns:
        --------
        str
            AI response to user question
        """
        
        try:
            # Build context for the question
            context = {
                'user_question': user_question,
                'performance_data': self._format_performance_data(performance_metrics),
                'config_summary': {
                    'num_assets': len(config.get('final_tickers', [])),
                    'time_period': f"{config.get('start_year')}-{config.get('end_year')}",
                    'estimation_window': f"{config.get('estimation_window')} months",
                    'constraints': config.get('constraints', {}),
                    'risk_free_rate': f"{config.get('risk_free_rate', 0.042):.1%}"
                }
            }
            
            # Add additional context if provided
            if context_data:
                context.update(context_data)
                
            prompt = self.prompts.get_chat_prompt()
            
            if isinstance(self.llm, ChatOpenAI):
                messages = [
                    SystemMessage(content="You are a quantitative finance expert helping interpret portfolio optimization results. Provide accurate, educational responses about the analysis."),
                    HumanMessage(content=prompt.format(**context))
                ]
                response = self.llm.invoke(messages)
                return response.content
            else:
                chain = prompt | self.llm | StrOutputParser()
                response = chain.invoke(context)
                return response
                
        except Exception as e:
            self.logger.error(f"Error in chat response: {e}")
            return f"I'm having trouble processing your question. Error: {str(e)}"