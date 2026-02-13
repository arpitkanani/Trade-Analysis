import streamlit as st  # type: ignore
import pandas as pd
import joblib # type: ignore    
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("📊 Hyperliquid Trader Behavior Dashboard")

# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    sentiment = pd.read_csv("data/fear_greed_index.csv")
    trades = pd.read_csv("data/historical_data.csv")

    sentiment['date'] = pd.to_datetime(sentiment['date'])

    trades['Timestamp IST'] = pd.to_datetime(
        trades['Timestamp IST'],
        format="%d-%m-%Y %H:%M",
        errors='coerce'
    )
    trades['date'] = trades['Timestamp IST'].dt.floor('D') # type: ignore

    df = trades.merge(
        sentiment[['date', 'classification']],
        on='date',
        how='inner'
    )

    return df





# ============================================
# LOAD MODEL OBJECTS
# ============================================
@st.cache_resource
def load_objects():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler_model.pkl")
    le = joblib.load("models/label_en.pkl")
    return model, scaler, le

model, scaler, le = load_objects()





df = load_data()

# ---------------------------
# Feature Engineering
# ---------------------------
df['is_win'] = (df['Closed PnL'] > 0).astype(int)
df['is_long'] = (df['Side'].str.upper() == 'BUY').astype(int)
df['abs_size_usd'] = df['Size USD'].abs()

daily = df.groupby(
    ['date', 'Account', 'classification']
).agg(
    daily_pnl=('Closed PnL', 'sum'),
    trades_per_day=('Trade ID', 'count'),
    avg_trade_size_usd=('abs_size_usd', 'mean'),
    win_rate=('is_win', 'mean'),
    long_ratio=('is_long', 'mean')
).reset_index()

# ---------------------------
# Sentiment encoding
# ---------------------------
sentiment_map = {
    'Extreme Fear': -2,
    'Fear': -1,
    'Neutral': 0,
    'Greed': 1,
    'Extreme Greed': 2
}

daily['sentiment_encoded'] = daily['classification'].map(sentiment_map)

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filters")
sentiment_filter = st.sidebar.multiselect(
    "Select Sentiment",
    daily['classification'].unique(),
    default=daily['classification'].unique()
)

filtered = daily[daily['classification'].isin(sentiment_filter)]

# ---------------------------
# SECTION 1: Performance
# ---------------------------
st.header("📈 Performance by Market Sentiment")

performance = filtered.groupby('classification').agg(
    avg_daily_pnl=('daily_pnl', 'mean'),
    win_rate=('win_rate', 'mean'),
    drawdown_proxy=('daily_pnl', lambda x: (x < 0).mean())
)

st.dataframe(performance)

st.bar_chart(performance['avg_daily_pnl'])
st.bar_chart(performance['win_rate'])
st.bar_chart(performance['drawdown_proxy'])

# ---------------------------
# SECTION 2: Behavior
# ---------------------------
st.header("🧠 Trader Behavior by Sentiment")

behavior = filtered.groupby('classification').agg(
    avg_trades_per_day=('trades_per_day', 'mean'),
    avg_trade_size=('avg_trade_size_usd', 'mean'),
    avg_long_ratio=('long_ratio', 'mean')
)

st.dataframe(behavior)

st.bar_chart(behavior['avg_trades_per_day'])
st.bar_chart(behavior['avg_trade_size'])
st.bar_chart(behavior['avg_long_ratio'])

# ---------------------------
# SECTION 3: Trader Segments
# ---------------------------
st.header("👥 Trader Segmentation")

# Frequency
trade_counts = df.groupby('Account').size()
median_trades = trade_counts.median()

df['frequency_group'] = df['Account'].map(
    lambda x: 'Frequent' if trade_counts[x] >= median_trades else 'Infrequent'
)

freq_segment = df.groupby('frequency_group').agg(
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('is_win', 'mean')
)

st.subheader("Frequent vs Infrequent Traders")
st.dataframe(freq_segment)
st.bar_chart(freq_segment)

# Consistency
pnl_std = df.groupby('Account')['Closed PnL'].std()
median_std = pnl_std.median()

df['consistency_group'] = df['Account'].map(
    lambda x: 'Consistent' if pnl_std[x] <= median_std else 'Inconsistent'
)

consistency_segment = df.groupby('consistency_group').agg(
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('is_win', 'mean')
)

st.subheader("Consistent vs Inconsistent Traders")
st.dataframe(consistency_segment)
st.bar_chart(consistency_segment)

# ---------------------------
# SECTION 4: Clustering
# ---------------------------
st.header("🔍 Behavioral Archetypes (Clustering)")

features = [
    'sentiment_encoded',
    'trades_per_day',
    'avg_trade_size_usd',
    'long_ratio',
    'win_rate'
]

X = daily[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Select number of clusters", 2, 5, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
daily.loc[X.index, 'cluster'] = kmeans.fit_predict(X_scaled)

cluster_summary = daily.groupby('cluster').agg(
    avg_trades_per_day=('trades_per_day', 'mean'),
    avg_trade_size=('avg_trade_size_usd', 'mean'),
    avg_win_rate=('win_rate', 'mean'),
    avg_pnl=('daily_pnl', 'mean')
)

st.dataframe(cluster_summary)
st.bar_chart(cluster_summary)
st.header("🔮 Next-Day Profitability Prediction")

st.write(
    "Enter trader behavior metrics and market sentiment to predict "
    "whether the trader is likely to be profitable the next day."
)


# ============================================
# USER INPUTS (RAW)
# ============================================
win_rate = st.slider("Win Rate", 0.0, 1.0, 0.5)
avg_trade_size_usd = st.number_input("Average Trade Size (USD)", 0.0, value=5000.0)
trades_per_day = st.number_input("Trades per Day", min_value=1, value=10)
long_ratio = st.slider("Long Ratio (BUY %)", 0.0, 1.0, 0.5)

classification = st.selectbox(
    "Market Sentiment",
    le.classes_
)

# ============================================
# PREDICTION
# ============================================
if st.button("Predict Next Day"):

    # 1️⃣ Encode sentiment
    sentiment_encoded = le.transform([classification])[0]

    # 2️⃣ Create raw input (order DOES NOT matter here)
    input_df = pd.DataFrame([{
        'win_rate': win_rate,
        'avg_trade_size_usd': avg_trade_size_usd,
        'trades_per_day': trades_per_day,
        'long_ratio': long_ratio,
        'sentiment_encoded': sentiment_encoded
    }])

    # 3️⃣ FORCE correct column order (THIS IS THE FIX)
    input_df = input_df[scaler.feature_names_in_]

    # 4️⃣ Scale
    input_scaled = scaler.transform(input_df)

    # 5️⃣ Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.success(f"✅ PROFITABLE next day (prob = {prob:.2f})")
    else:
        st.error(f"❌ NOT profitable next day (prob = {prob:.2f})")

# ---------------------------
st.success("Dashboard ready. Explore insights interactively 🚀")
