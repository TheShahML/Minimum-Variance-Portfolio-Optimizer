# src/ai/prompts.py
"""
LLM prompts for portfolio analysis and commentary
"""

from langchain_core.prompts import PromptTemplate

class PortfolioPrompts:
    """Collection of prompts for portfolio analysis"""
    
    def __init__(self):
        self.base_disclaimer = """
        
        IMPORTANT DISCLAIMER: This analysis is for educational purposes only and does not constitute investment advice. 
        Past performance does not guarantee future results. All investments carry risk of loss.
        """
    
    def get_performance_summary_prompt(self) -> PromptTemplate:
        """Prompt for generating performance summary"""
        
        template = """
        As a quantitative finance expert, provide a comprehensive analysis of this portfolio optimization backtest:

        PORTFOLIO CONFIGURATION:
        - Number of Assets: {num_assets}
        - Time Period: {time_period}
        - Estimation Window: {estimation_window}
        - Risk-Free Rate: {risk_free_rate}
        - Constraints: {constraints}
        
        PERFORMANCE RESULTS:
        {performance_data}
        
        Please provide:
        1. Executive Summary: Key findings in 2-3 sentences
        2. Performance Comparison: Which method (Sample Covariance vs Ledoit-Wolf) performed better and by how much
        3. Risk-Return Profile: Analysis of volatility vs returns achieved
        4. Statistical Significance: Commentary on the reliability of the results given the sample size
        5. Market Context: What these results suggest about market efficiency during this period
        
        Focus on:
        - Practical insights for portfolio management
        - Statistical rigor and methodology validation
        - Educational explanations of why certain results occurred
        
        Keep the analysis objective and educational.{base_disclaimer}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "performance_data", "num_assets", "time_period", 
                "estimation_window", "risk_free_rate", "constraints", "base_disclaimer"
            ],
            partial_variables={"base_disclaimer": self.base_disclaimer}
        )
    
    def get_methodology_explanation_prompt(self) -> PromptTemplate:
        """Prompt for explaining optimization methodology"""
        
        template = """
        Explain the portfolio optimization methodology and interpret the results for an educated audience:

        METHODOLOGY DETAILS:
        - Optimization Objective: Minimum variance portfolio construction
        - Methods Compared: Sample Covariance vs Ledoit-Wolf Shrinkage Estimation
        - Rolling Window: {estimation_window} months
        - Universe: {num_assets} assets over {time_period}
        - Constraints: {constraints_info}
        
        RESULTS SUMMARY:
        - Better Performing Method: {better_method}
        - Sample Covariance Sharpe Ratio: {sample_sharpe}
        - Ledoit-Wolf Sharpe Ratio: {lw_sharpe}
        - Performance Difference: {sharpe_difference}
        
        SHRINKAGE INFORMATION:
        {shrinkage_info}
        
        Please explain:
        1. Why minimum variance optimization matters in portfolio management
        2. The theoretical advantage of Ledoit-Wolf shrinkage over sample covariance
        3. When each method tends to work better (market conditions, portfolio size, etc.)
        4. Why the observed performance difference occurred in this specific case
        5. The role of the rolling window approach in avoiding look-ahead bias
        6. Practical implications for real-world portfolio management
        
        Make this accessible to someone with basic finance knowledge but detailed enough to be educational.{base_disclaimer}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "better_method", "sharpe_difference", "sample_sharpe", "lw_sharpe",
                "estimation_window", "num_assets", "time_period", "shrinkage_info",
                "constraints_info", "base_disclaimer"
            ],
            partial_variables={"base_disclaimer": self.base_disclaimer}
        )
    
    def get_portfolio_commentary_prompt(self) -> PromptTemplate:
        """Prompt for portfolio composition commentary"""
        
        template = """
        Analyze the portfolio composition and positioning strategy:

        PORTFOLIO WEIGHTS (FINAL PERIOD):
        {portfolio_weights}
        
        CONCENTRATION METRICS:
        {concentration_metrics}
        
        PORTFOLIO CHARACTERISTICS:
        - Total Assets: {num_assets}
        - Analysis Period: {time_period}
        - Rebalancing: {rebalance_frequency}
        - Constraints: {constraints}
        
        PERFORMANCE COMPARISON:
        {performance_comparison}
        
        Please analyze:
        1. Portfolio Concentration: Is the portfolio well-diversified or concentrated? What are the implications?
        2. Position Sizing: Comment on the largest positions and whether they seem reasonable
        3. Diversification Effectiveness: How well does the portfolio spread risk across assets?
        4. Sector/Style Biases: Any observable patterns in the weightings (if sector info available)
        5. Turnover Implications: How might the weight distribution affect trading costs?
        6. Risk Budgeting: How are the positions contributing to overall portfolio risk?
        
        Focus on practical portfolio management insights and risk considerations.{base_disclaimer}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "portfolio_weights", "concentration_metrics", "num_assets", 
                "performance_comparison", "time_period", "rebalance_frequency",
                "constraints", "base_disclaimer"
            ],
            partial_variables={"base_disclaimer": self.base_disclaimer}
        )
    
    def get_risk_analysis_prompt(self) -> PromptTemplate:
        """Prompt for risk analysis commentary"""
        
        template = """
        Provide a comprehensive risk analysis of the portfolio optimization results:

        RISK METRICS BY METHOD:
        {risk_metrics}
        
        TURNOVER ANALYSIS:
        {turnover_metrics}
        
        PORTFOLIO PARAMETERS:
        - Estimation Window: {estimation_window}
        - Time Period: {time_period}
        - Risk-Free Rate: {risk_free_rate}
        - Constraints: {constraints}
        
        Analyze the following risk dimensions:
        
        1. VOLATILITY ANALYSIS:
           - Compare annualized volatility between methods
           - Assess if volatility levels are appropriate for minimum variance objectives
           
        2. DOWNSIDE RISK:
           - Interpret maximum drawdown figures
           - Analyze tail risk characteristics (skewness, kurtosis)
           - Comment on worst-case scenarios
           
        3. PORTFOLIO TURNOVER:
           - Assess trading frequency and implementation costs
           - Discuss stability of optimization results
           - Consider market impact of rebalancing
           
        4. RISK-ADJUSTED PERFORMANCE:
           - Interpret Sharpe and Sortino ratios
           - Assess consistency of risk-adjusted returns
           
        5. IMPLEMENTATION RISKS:
           - Transaction costs implications
           - Liquidity considerations
           - Model risk and parameter uncertainty
        
        Provide actionable insights for risk management and portfolio implementation.{base_disclaimer}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "risk_metrics", "turnover_metrics", "estimation_window", 
                "time_period", "risk_free_rate", "constraints", "base_disclaimer"
            ],
            partial_variables={"base_disclaimer": self.base_disclaimer}
        )
    
    def get_recommendations_prompt(self) -> PromptTemplate:
        """Prompt for investment recommendations"""
        
        template = """
        Based on the portfolio optimization analysis, provide actionable investment recommendations:

        ANALYSIS RESULTS:
        - Superior Method: {better_method}
        - Sample Covariance Sharpe Ratio: {sample_sharpe} (Volatility: {sample_volatility})
        - Ledoit-Wolf Sharpe Ratio: {lw_sharpe} (Volatility: {lw_volatility})
        - Analysis Period: {time_period}
        - Portfolio Size: {num_assets} assets
        - Estimation Window: {estimation_window} months
        - Current Constraints: {current_constraints}
        - Analysis Type: {analysis_type}
        
        Provide specific recommendations in these areas:
        
        1. METHODOLOGY SELECTION:
           - Which covariance estimation method to use going forward and why
           - When to consider switching between methods
        
        2. IMPLEMENTATION PARAMETERS:
           - Optimal rebalancing frequency considering turnover costs
           - Suggested estimation window adjustments
           - Portfolio constraints modifications
        
        3. PORTFOLIO ENHANCEMENTS:
           - Additional risk factors to consider
           - Potential improvements to the optimization framework
           - Universe expansion or contraction recommendations
        
        4. RISK MANAGEMENT:
           - Key risks to monitor
           - Early warning indicators
           - Stress testing suggestions
        
        5. NEXT STEPS:
           - Specific actions for implementation
           - Additional analysis to conduct
           - Performance monitoring framework
        
        Structure recommendations as actionable steps with clear reasoning.{base_disclaimer}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "better_method", "sample_sharpe", "lw_sharpe", "sample_volatility",
                "lw_volatility", "time_period", "num_assets", "estimation_window",
                "current_constraints", "analysis_type", "base_disclaimer"
            ],
            partial_variables={"base_disclaimer": self.base_disclaimer}
        )
    
    def get_chat_prompt(self) -> PromptTemplate:
        """Prompt for interactive chat about results"""
        
        template = """
        You are helping interpret portfolio optimization results. Answer the user's question accurately and educationally.

        USER QUESTION: {user_question}

        PORTFOLIO ANALYSIS CONTEXT:
        Performance Data: {performance_data}
        Configuration: {config_summary}

        Guidelines for your response:
        - Be accurate and educational
        - Reference specific data points when relevant
        - Explain concepts clearly
        - Acknowledge limitations of the analysis
        - Suggest follow-up analysis if appropriate
        - Keep responses concise but informative
        
        If the question is outside the scope of the provided data, clearly state what information you don't have access to.{base_disclaimer}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "user_question", "performance_data", "config_summary", "base_disclaimer"
            ],
            partial_variables={"base_disclaimer": self.base_disclaimer}
        )