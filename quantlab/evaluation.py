"""
Evaluation and performance metrics module.
"""

import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Evaluator:
    """
    Evaluate trading strategy performance.
    """
    
    def __init__(self):
        """Initialize Evaluator."""
        pass
    
    def calculate_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: Series of equity values over time
            
        Returns:
            Dictionary of metrics
        """
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annualized_return = (1 + total_return / 100) ** (252 / len(returns)) - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio
        sharpe_ratio = (annualized_return / (volatility / 100)) if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return / (downside_std / 100)) if downside_std > 0 else 0
        
        # Max Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Win Rate (from returns)
        winning_trades = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Average R/R
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return * 100,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_rr': avg_rr,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades
        }
        
        logger.info(f"Metrics calculated - Sharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2f}%")
        
        return metrics
    
    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve.
        
        Args:
            equity_curve: Series of equity values
            title: Plot title
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve.index, equity_curve.values, linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=equity_curve.iloc[0], color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_interactive(
        self,
        df: pd.DataFrame,
        equity_curve: pd.Series,
        price_col: str = 'close',
        save_path: Optional[str] = None
    ):
        """
        Create interactive plot with price and equity curve.
        
        Args:
            df: DataFrame with price data
            equity_curve: Series of equity values
            price_col: Column name for price
            save_path: Path to save HTML
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price', 'Equity Curve', 'Drawdown'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price
        fig.add_trace(
            go.Scatter(x=df.index, y=df[price_col], name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Equity Curve
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve.values, name='Equity', line=dict(color='green')),
            row=2, col=1
        )
        
        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, name='Drawdown', fill='tonexty', line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(
            height=900,
            title_text="Trading Strategy Performance",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")
        
        fig.show()
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        equity_curve: pd.Series,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate performance report.
        
        Args:
            metrics: Dictionary of metrics
            equity_curve: Series of equity values
            save_path: Path to save report
            
        Returns:
            Report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           TRADING STRATEGY PERFORMANCE REPORT                ║
╚══════════════════════════════════════════════════════════════╝

📊 RETURNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Return:          {metrics['total_return']:>10.2f}%
Annualized Return:     {metrics['annualized_return']:>10.2f}%
Volatility:            {metrics['volatility']:>10.2f}%

📈 RISK METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}
Sortino Ratio:         {metrics['sortino_ratio']:>10.2f}
Max Drawdown:          {metrics['max_drawdown']:>10.2f}%

💰 TRADING STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades:          {metrics['total_trades']:>10.0f}
Winning Trades:        {metrics['winning_trades']:>10.0f}
Losing Trades:         {metrics['losing_trades']:>10.0f}
Win Rate:              {metrics['win_rate']:>10.2f}%
Profit Factor:         {metrics['profit_factor']:>10.2f}
Average R/R:           {metrics['average_rr']:>10.2f}

📅 PERIOD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start Date:            {equity_curve.index[0].strftime('%Y-%m-%d'):>10}
End Date:              {equity_curve.index[-1].strftime('%Y-%m-%d'):>10}
Duration:              {len(equity_curve):>10} bars

╚══════════════════════════════════════════════════════════════╝
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        print(report)
        return report


from typing import Optional, Dict

