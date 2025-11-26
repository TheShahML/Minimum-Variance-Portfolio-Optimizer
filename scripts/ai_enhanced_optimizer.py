"""
Portfolio Optimization Script - AI-Enhanced Version
Modularized version with AI-powered portfolio analysis and recommendations
Uses interactive inputs with comprehensive validation + AI insights
"""

import os
import sys
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging to see detailed error messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio.optimizer import PortfolioOptimizer
from portfolio.data_handler import DataHandler
from utils.plotting import PortfolioPlotter
from utils.validation import InputValidator
from ai.langchain_analyzer import LangChainPortfolioAnalyzer
from ai.local_ai_analyzer import LocalAIPortfolioAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_user_inputs():
    """
    Interactive input collection with validation
    Matches the original notebook's get_user_inputs() function
    """
    print("\n" + "=" * 70)
    print("MINIMUM VARIANCE PORTFOLIO OPTIMIZATION")
    print("=" * 70)

    # STEP 1: Connect to WRDS
    print("\nSTEP 1: Connecting to WRDS")
    print("-" * 50)
    print("Establishing database connection...")

    data_handler = DataHandler()

    # This will prompt for WRDS username/password if needed
    try:
        db_connection = data_handler.connect_to_wrds()
        if db_connection is None:
            print("ERROR: Could not connect to WRDS")
            return None
        print("Connection established")
    except Exception as e:
        print(f"ERROR: WRDS connection failed - {e}")
        return None

    default_tickers = data_handler.get_default_tickers()
    validator = InputValidator()

    # STEP 2: Backtest Period
    print("\nSTEP 2: Backtest Period")
    print("-" * 50)

    while True:
        try:
            start_year_input = input("Start year (default 2010): ") or "2010"
            start_year = int(start_year_input)
            if start_year < 2000:
                print("ERROR: Start year must be 2000 or later")
                continue
            break
        except ValueError:
            print("ERROR: Please enter a valid year")

    while True:
        try:
            end_year_input = input("End year (default 2024): ") or "2024"
            end_year = int(end_year_input)
            if end_year <= start_year:
                print(f"ERROR: End year must be after {start_year}")
                continue
            if end_year - start_year < 2:
                print("ERROR: Need at least 2 years of data")
                continue
            break
        except ValueError:
            print("ERROR: Please enter a valid year")

    print(f" Period: {start_year}-{end_year} ({end_year - start_year} years)")

    # STEP 3: Stock Selection
    print("\nSTEP 3: Stock Selection")
    print("-" * 50)
    print(f"Enter tickers (space-separated) or press Enter for default {len(default_tickers)} blue-chip stocks")

    while True:
        user_input = input("Tickers: ").strip().upper()

        if user_input:
            # Remove duplicates while preserving order
            raw_tickers = user_input.split()
            tickers = []
            seen = set()
            duplicates = []

            for ticker in raw_tickers:
                if ticker in seen:
                    duplicates.append(ticker)
                else:
                    tickers.append(ticker)
                    seen.add(ticker)

            # Warn about duplicates
            if duplicates:
                print(f"WARNING: Removed duplicate tickers: {', '.join(duplicates)}")
                print(f"Using {len(tickers)} unique tickers: {', '.join(tickers)}")

            # Check minimum ticker count
            if len(tickers) < 10:
                print(f"ERROR: Only {len(tickers)} unique tickers provided.")
                print("Assignment requires at least 10 unique stocks for meaningful analysis.")
                print("Please enter at least 10 unique tickers or press Enter for defaults:")
                continue

            # Validate tickers by fetching data
            print(f"Validating {len(tickers)} unique tickers from {start_year} to {end_year}...")
            temp_start = f"{start_year}-01-01"
            temp_end = f"{end_year}-12-31"
            temp_data = data_handler.fetch_stock_returns(tickers, temp_start, temp_end, db_connection)

            if temp_data is None or temp_data.empty:
                print("ERROR: No data found for these tickers in the specified period.")
                print("Please enter different tickers or press Enter for defaults:")
                continue

            # Check for insufficient data coverage
            missing_tickers = []
            period_years = end_year - start_year
            min_required_obs = period_years * 12 * 0.75  # Require 75% data coverage

            for ticker in tickers:
                if ticker not in temp_data.columns:
                    missing_tickers.append(f"{ticker} (not found)")
                elif temp_data[ticker].notna().sum() < min_required_obs:
                    coverage = temp_data[ticker].notna().sum() / len(temp_data) * 100
                    missing_tickers.append(f"{ticker} ({coverage:.0f}% coverage)")

            if missing_tickers:
                print(f"ERROR: These tickers have insufficient data:")
                for ticker in missing_tickers:
                    print(f"  - {ticker}")
                print("Please enter different tickers or press Enter for defaults:")
                continue

            print(" All tickers have sufficient data")
            print(f" Using {len(tickers)} custom tickers")
            break
        else:
            tickers = default_tickers
            print(f" Using {len(tickers)} default tickers")
            break

    # STEP 4: Estimation Window
    print("\nSTEP 4: Estimation Window")
    print("-" * 50)

    while True:
        try:
            est_window_input = input("Estimation window in months (default 36): ") or "36"
            estimation_window = int(est_window_input)
            if estimation_window < 6:
                print("ERROR: Must be at least 6 months")
                continue
            if estimation_window > 120:
                print("WARNING: Very long window (>10 years)")
                if input("Continue? (y/n): ").lower() != 'y':
                    continue
            break
        except ValueError:
            print("ERROR: Please enter a valid number")

    print(f" Window: {estimation_window} months")

    # STEP 5: Portfolio Constraints
    print("\nSTEP 5: Portfolio Constraints")
    print("-" * 50)

    while True:
        allow_short = input("Allow short positions? (y/n, default n): ").lower().strip()
        if allow_short in ['y', 'yes', 'n', 'no', '']:
            break
        print("Please enter 'y' or 'n'")

    allow_short = allow_short in ['y', 'yes']

    if allow_short:
        print("Long/short portfolio selected")
        min_weight = -1.0
        max_weight = 1.0
    else:
        print("Long-only portfolio selected")
        min_weight = 0.0

        while True:
            use_max = input("Limit maximum weight per stock? (y/n, default y): ").lower().strip()
            if use_max in ['y', 'yes', 'n', 'no', '']:
                break

        if use_max in ['', 'y', 'yes']:
            while True:
                try:
                    max_input = input("Maximum weight (e.g., 0.25 for 25%, default 0.25): ") or "0.25"
                    max_weight = float(max_input)
                    if max_weight <= 0 or max_weight > 1:
                        print("ERROR: Must be between 0 and 1")
                        continue
                    break
                except ValueError:
                    print("ERROR: Please enter a valid number")
        else:
            max_weight = 1.0

    print(f" Constraints: [{min_weight}, {max_weight}], shorts={'allowed' if allow_short else 'not allowed'}")

    # STEP 6: Risk-Free Rate
    print("\nSTEP 6: Risk-Free Rate")
    print("-" * 50)

    while True:
        try:
            rf_input = input("Annual risk-free rate % (default 4.2): ") or "4.2"
            risk_free_rate = float(rf_input) / 100
            if risk_free_rate < 0:
                print("ERROR: Must be positive")
                continue
            if risk_free_rate > 0.20:
                print("WARNING: >20% seems very high")
                if input("Continue? (y/n): ").lower() != 'y':
                    continue
            break
        except ValueError:
            print("ERROR: Please enter a valid number")

    print(f" Risk-free rate: {risk_free_rate:.2%}")

    # STEP 7: Data Quality
    print("\nSTEP 7: Data Quality Requirements")
    print("-" * 50)

    while True:
        try:
            min_obs_input = input(f"Min observations per stock (default {estimation_window}): ") or str(estimation_window)
            min_observations = int(min_obs_input)
            if min_observations < 1:
                print("ERROR: Must be at least 1")
                continue
            break
        except ValueError:
            print("ERROR: Please enter a valid number")

    while True:
        try:
            max_missing_input = input("Max missing data % (default 10): ") or "10"
            max_missing_pct = float(max_missing_input) / 100
            if max_missing_pct < 0 or max_missing_pct > 1:
                print("ERROR: Must be between 0 and 100")
                continue
            break
        except ValueError:
            print("ERROR: Please enter a valid number")

    print(f" Data quality: min_obs={min_observations}, max_missing={max_missing_pct:.0%}")

    # Build config
    config = {
        'tickers': tickers,
        'start_year': start_year,
        'end_year': end_year,
        'estimation_window': estimation_window,
        'constraints': {
            'min_weight': min_weight,
            'max_weight': max_weight,
            'allow_short': allow_short,
            'long_only': not allow_short
        },
        'risk_free_rate': risk_free_rate,
        'coverage_params': {
            'min_observations': min_observations,
            'max_missing_pct': max_missing_pct
        },
        'db_connection': db_connection
    }

    # Final validation
    print("\n" + "-" * 70)
    print("VALIDATING CONFIGURATION...")
    validated_config = validator.validate_optimization_config(config)
    print(" All inputs validated successfully!")

    return validated_config
def print_results(results):
    """Display performance results"""
    perf = results.get('performance_metrics', {})

    if 'sample' in perf and 'lw' in perf:
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)

        print(f"\n{'METRIC':<30} {'SAMPLE':<15} {'LEDOIT-WOLF':<15}")
        print("-" * 70)

        metrics = [
            ('Total Return', 'total_return', '.2%'),
            ('Annualized Return', 'annualized_return', '.2%'),
            ('Annualized Volatility', 'annualized_volatility', '.2%'),
            ('Sharpe Ratio', 'sharpe_ratio', '.3f'),
            ('Sortino Ratio', 'sortino_ratio', '.3f'),
            ('Max Drawdown', 'max_drawdown', '.2%'),
            ('Win Rate', 'win_rate', '.1%'),
        ]

        for name, key, fmt in metrics:
            sample_val = perf['sample'].get(key, 0)
            lw_val = perf['lw'].get(key, 0)
            sample_str = f"{sample_val:{fmt}}"
            lw_str = f"{lw_val:{fmt}}"
            print(f"{name:<30} {sample_str:<15} {lw_str:<15}")

        sample_sharpe = perf['sample'].get('sharpe_ratio', 0)
        lw_sharpe = perf['lw'].get('sharpe_ratio', 0)

        print("-" * 70)
        if lw_sharpe > sample_sharpe:
            improvement = (lw_sharpe - sample_sharpe) / sample_sharpe * 100 if sample_sharpe != 0 else 0
            print(f"WINNER: Ledoit-Wolf Shrinkage (+{improvement:.1f}% Sharpe improvement)")
        else:
            improvement = (sample_sharpe - lw_sharpe) / lw_sharpe * 100 if lw_sharpe != 0 else 0
            print(f"WINNER: Sample Covariance (+{improvement:.1f}% Sharpe improvement)")


def save_results(results, config):
    """Save results to results folder"""
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save backtest results
    if results.get('backtest_results') is not None:
        filepath = os.path.join(results_dir, f"backtest_results_{timestamp}.csv")
        results['backtest_results'].to_csv(filepath)
        print(f" Saved: {filepath}")

    # Save portfolio weights
    if results.get('portfolio_weights') is not None:
        filepath = os.path.join(results_dir, f"portfolio_weights_{timestamp}.csv")
        results['portfolio_weights'].to_csv(filepath)
        print(f" Saved: {filepath}")

    # Save performance summary
    filepath = os.path.join(results_dir, f"performance_summary_{timestamp}.txt")
    with open(filepath, 'w') as f:
        f.write("MINIMUM VARIANCE PORTFOLIO OPTIMIZATION - PERFORMANCE SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Period: {config['start_year']}-{config['end_year']}\n")
        f.write(f"Tickers: {len(config['tickers'])} stocks\n")
        f.write(f"Estimation Window: {config['estimation_window']} months\n")
        f.write(f"Risk-Free Rate: {config['risk_free_rate']:.2%}\n\n")

        if 'performance_metrics' in results:
            perf = results['performance_metrics']
            f.write("PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'METRIC':<30} {'SAMPLE':<15} {'LEDOIT-WOLF':<15}\n")

            metrics = [
                ('Annualized Return', 'annualized_return', '.2%'),
                ('Annualized Volatility', 'annualized_volatility', '.2%'),
                ('Sharpe Ratio', 'sharpe_ratio', '.3f'),
                ('Max Drawdown', 'max_drawdown', '.2%')
            ]

            for name, key, fmt in metrics:
                sample_val = perf['sample'].get(key, 0)
                lw_val = perf['lw'].get(key, 0)
                # Format first, then align
                sample_str = f"{sample_val:{fmt}}"
                lw_str = f"{lw_val:{fmt}}"
                f.write(f"{name:<30} {sample_str:<15} {lw_str:<15}\n")

    print(f" Saved: {filepath}")


def generate_visualizations(results, optimizer, save_figures=False):
    """Generate plots"""
    print("\nGenerating visualizations...")

    plotter = PortfolioPlotter(style='modern')
    figures = []

    try:
        # Performance dashboard
        fig1 = plotter.create_performance_dashboard(
            results['backtest_results'],
            results['portfolio_weights'],
            results['performance_metrics']
        )
        figures.append(('performance_dashboard', fig1))
        plt.show(block=False)

        # Efficient frontier
        fig2 = plotter.plot_efficient_frontier_comparison(
            optimizer.returns_data,
            results['config']
        )
        figures.append(('efficient_frontier', fig2))
        plt.show(block=False)

        # Summary table
        fig3 = plotter.create_summary_table(
            results['portfolio_weights'],
            results['performance_metrics'],
            results['config']['final_tickers']
        )
        figures.append(('summary_table', fig3))
        plt.show(block=False)

        print(" All visualizations displayed")
        
        # Ask if user wants to save figures
        save_viz = input("\nSave visualizations to file? (y/n): ").lower().strip()
        if save_viz in ['y', 'yes']:
            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for name, fig in figures:
                filepath = os.path.join(results_dir, f"{name}_{timestamp}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f" Saved: {filepath}")
        
        print("Close plot windows to continue...")
        plt.show(block=True)

    except Exception as e:
        print(f" Visualization error: {e}")


def main():
    """Main entry point"""
    try:
        # Get user inputs
        config = get_user_inputs()

        if config is None:
            print("Configuration cancelled or failed")
            return

        # Run optimization
        print("\n" + "=" * 70)
        print("RUNNING OPTIMIZATION")
        print("-" * 70)
        print("Running backtest with rolling window...")
        print("This may take a few minutes...\n")

        # Use the existing WRDS connection from config
        db_connection = config.get('db_connection')

        optimizer = PortfolioOptimizer(
            tickers=config['tickers'],
            start_year=config['start_year'],
            end_year=config['end_year'],
            estimation_window=config['estimation_window'],
            constraints=config['constraints'],
            risk_free_rate=config['risk_free_rate'],
            coverage_params=config['coverage_params']
        )

        results = optimizer.run_complete_analysis(db_connection=db_connection)

        # Close database connection after analysis
        if db_connection is not None:
            try:
                db_connection.close()
                print("\nDatabase connection closed")
            except:
                pass

        if results['success']:
            print("\n Analysis completed successfully!")

            # Display results
            print_results(results)

            # AI Analysis
            print("\n" + "=" * 70)
            print("AI-POWERED ANALYSIS")
            print("-" * 70)
            ai = input("Generate AI analysis and investment recommendations? (y/n): ").lower().strip()

            if ai in ['y', 'yes']:
                try:
                    # Check if OpenAI key is available
                    has_openai_key = os.getenv('OPENAI_API_KEY') is not None and os.getenv('OPENAI_API_KEY').strip() != ''

                    if has_openai_key:
                        print("\n OpenAI API key detected")
                        use_openai = input("Use OpenAI GPT for analysis? (y/n, default n): ").lower().strip()
                        use_openai = use_openai in ['y', 'yes']
                    else:
                        print("\n No OpenAI API key found")
                        print("  To use OpenAI:")
                        print("  1. Add OPENAI_API_KEY to .env file")
                        print("  2. Add credits at https://platform.openai.com/billing")
                        print("\n  Using local models...")
                        use_openai = False

                    # Initialize appropriate analyzer
                    if use_openai:
                        print("\nInitializing OpenAI analyzer...")
                        ai_analyzer = LangChainPortfolioAnalyzer(use_openai=True)
                    else:
                        print("\nInitializing local AI analyzer...")
                        print("Note: Local models provide basic analysis")
                        print("For better results, configure OpenAI API key in .env file")

                        # Ask user preference
                        use_langchain = input("\nUse LangChain with local models? (y/n, default y): ").lower().strip()

                        if use_langchain in ['', 'y', 'yes']:
                            ai_analyzer = LangChainPortfolioAnalyzer(use_openai=False)
                        else:
                            ai_analyzer = LocalAIPortfolioAnalyzer()

                    print("\nGenerating comprehensive AI analysis...")
                    ai_report = ai_analyzer.generate_complete_report(
                        performance_metrics=results['performance_metrics'],
                        portfolio_weights=results.get('portfolio_weights', {}),
                        turnover_metrics=results.get('turnover_metrics', {}),
                        config=results.get('config', {})
                    )

                    print("\n" + "=" * 70)
                    print("AI ANALYSIS REPORT")
                    print("=" * 70)

                    for section_name, content in ai_report.items():
                        # Skip metadata fields
                        if section_name not in ['generated_at', 'model_used', 'device', 'error']:
                            print(f"\n{section_name.upper().replace('_', ' ')}:")
                            print("-" * 40)
                            print(content.strip())
                            print()

                    print("=" * 70)

                    # Option to save AI report
                    save_ai = input("\nSave AI report to file? (y/n): ").lower().strip()
                    if save_ai in ['y', 'yes']:
                        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                        os.makedirs(results_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filepath = os.path.join(results_dir, f"ai_analysis_{timestamp}.txt")

                        with open(filepath, 'w') as f:
                            f.write("AI-POWERED PORTFOLIO ANALYSIS REPORT\n")
                            f.write("=" * 70 + "\n\n")
                            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Model: {ai_report.get('model_used', 'Unknown')}\n\n")

                            for section_name, content in ai_report.items():
                                if section_name not in ['generated_at', 'model_used', 'device', 'error']:
                                    f.write(f"{section_name.upper().replace('_', ' ')}:\n")
                                    f.write("-" * 40 + "\n")
                                    f.write(content.strip() + "\n\n")

                        print(f" Saved AI report: {filepath}")

                except Exception as e:
                    print(f" AI analysis error: {e}")
                    print("Continuing without AI analysis...")

            # Visualizations
            print("\n" + "=" * 70)
            viz = input("Generate visualizations? (y/n): ").lower().strip()
            if viz in ['y', 'yes']:
                generate_visualizations(results, optimizer)

            # Save results
            print("\n" + "=" * 70)
            save = input("Save numerical results to file? (y/n): ").lower().strip()
            if save in ['y', 'yes']:
                save_results(results, config)

        else:
            print("\n Analysis failed:")
            for error in results.get('errors', ['Unknown error']):
                print(f"  - {error}")

        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        # Close connection on interrupt
        if 'config' in locals() and config is not None:
            db_conn = config.get('db_connection')
            if db_conn is not None:
                try:
                    db_conn.close()
                except:
                    pass
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        # Close connection on error
        if 'config' in locals() and config is not None:
            db_conn = config.get('db_connection')
            if db_conn is not None:
                try:
                    db_conn.close()
                except:
                    pass


if __name__ == "__main__":
    main()
