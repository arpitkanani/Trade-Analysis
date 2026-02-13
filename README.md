### Follow below step first to setup environment  of python

conda create -n venv python=3.10 -y

conda activate venv/         # Windows

pip install -r requirements.txt

---

## Methodology
This project analyzes historical trade-level data to understand how **market sentiment** (Extreme Fear,
Fear, Neutral, Greed, Extreme Greed) influences **trader performance and behavior**. Trades were 
aggregated at the **daily trader level** to compute key metrics such as daily PnL, win rate, trade 
frequency, average trade size (used as a leverage proxy), long/short ratio, and a drawdown proxy.  
Traders were further segmented into meaningful groups—**high vs low risk**, **frequent vs infrequent**, 
and **consistent vs inconsistent**—using statistically robust measures such as medians and PnL 
volatility. All insights were validated using grouped tables and distribution summaries.

---

## Insights

### Insights of Task 2
## Insight 1: Performance differs between Fear and Greed days  
**(Backed by: Performance-by-Sentiment table)**

- **Fear days** have higher **average PnL**, but also a higher **drawdown proxy**, indicating greater downside risk.
- **Greed days** show lower average PnL with **more stable outcomes**.
- **Win rate** is slightly higher during Fear than during Greed.

**Conclusion:**  
Fear regimes offer **higher reward at higher risk**, while Greed regimes are **more stable but less profitable**.

---

## Insight 2: Trader behavior changes significantly with sentiment  
**(Backed by: Behavior Summary & Daily Trader Metrics tables)**

- During **Fear / Extreme Fear**:
  - Traders execute **more trades per day**
  - Take **larger position sizes**
  - Exhibit a **stronger long bias** (buy-the-dip behavior)
- During **Greed / Extreme Greed**:
  - Fewer trades
  - Smaller average trade sizes
  - More balanced long/short exposure

**Conclusion:**  
Market sentiment primarily affects **how traders trade**, not just outcomes.

---

## Insight 3: Leverage risk is concentrated in Fear regimes  
**(Backed by: Leverage Proxy table – Size USD distribution)**

- **Fear** shows the highest:
  - Mean position size
  - Standard deviation
  - Maximum trade size (fat tails)
- Median sizes are similar across sentiments, implying risk is driven by **rare oversized trades**.

**Conclusion:**  
Systemic leverage risk spikes during Fear due to **extreme outlier positions**, not typical trades.

---

## Insight 4: Clear risk–return trade-offs across trader segments  
**(Backed by: Trader Segmentation tables)**

- **High-risk traders**:
  - Higher average PnL
  - Lower win rate
- **Low-risk traders**:
  - Higher win rate
  - Lower profitability
- **Frequent traders**:
  - More consistent wins
  - Lower PnL
- **Infrequent traders**:
  - Higher PnL
  - Lower consistency

**Conclusion:**  
Profitability is driven by **risk appetite and selectivity**, not win rate alone.

---


| Insight Area | Supporting Table / Chart | Key Observations | Conclusion |
|-------------|-------------------------|------------------|------------|
| **Performance by Sentiment (Fear vs Greed)** | Performance-by-Sentiment table (PnL, win rate, drawdown proxy) | Fear days show higher average PnL but also higher drawdown proxy; Greed days have lower PnL but more stable outcomes; win rate slightly higher in Fear | Fear regimes offer higher reward with higher risk, while Greed regimes are more stable but less profitable |
| **Behavior Change by Sentiment** | Behavior Summary & Daily Trader Metrics tables | Fear leads to more trades per day, larger position sizes, and stronger long bias; Greed leads to fewer trades, smaller sizes, and balanced exposure | Market sentiment significantly alters trading behavior rather than trader skill |
| **Leverage Risk Distribution** | Leverage Proxy table (Size USD distribution) | Fear has highest mean, std, and max position sizes; medians are similar across sentiments, indicating tail risk | Leverage risk is concentrated in Fear due to rare but extreme trades |
| **Trader Segmentation (Risk & Frequency)** | Risk-based & Frequency-based segmentation tables | High-risk and infrequent traders earn higher PnL but have lower win rates; low-risk and frequent traders show higher consistency | Profitability is driven by risk appetite and selectivity, not win rate alone |

--- 

## Strategy Recommendations
### Strategy 1:
- **Conviction-Based Trade Frequency Filter:** During Greed and Neutral days, restrict trade frequency for high-frequency traders and prioritize low-frequency, high-conviction trades with balanced long/short exposure.
#### Why this works:

- Frequent traders show higher win rates but lower PnL (overtrading effect).

- Infrequent traders deliver higher average PnL, especially in stable sentiment regimes.

- Greed days are less volatile, reducing the edge of rapid trading.

#### Actionable implementation:

- Set a maximum trades-per-day threshold in Greed regimes.
- Encourage fewer, larger, thesis-driven trades.
- Require long_ratio near 0.5 to avoid crowded directional bets.

- **Risk-controlled Fear strategy:** During Fear and Extreme Fear days, cap leverage for high-variance traders while allowing controlled scaling only for consistent traders to reduce tail losses.

#### Why this works:
- Fear days show highest leverage, largest tail risk, and higher drawdowns.
- Inconsistent traders generate profits through rare, oversized trades, which amplify downside risk.
- Consistent traders maintain higher win rates and better risk control.

#### Actionable implementation:
- Impose a leverage ceiling for inconsistent traders during Fear.
- Allow moderate size scaling only for traders with low PnL volatility.
- Objective: reduce tail losses without killing upside.