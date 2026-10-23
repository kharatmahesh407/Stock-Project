import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
 
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("📈 Stock Analysis Dashboard")
 
UP, DOWN = "#26a69a", "#ef5350"  # green / red like trading apps
TEMPLATE = "plotly_dark"
 
# ---------- Sidebar ----------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock symbol (NSE: .NS, BSE: .BO)", "RELIANCE.NS").upper()
ticker2 = st.sidebar.text_input("Compare with (optional)", "TCS.NS").upper()
period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
chart_type = st.sidebar.radio("Chart type", ["Candlestick", "Line"])
show_ma = st.sidebar.checkbox("Moving averages (20/50)", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=False)
 
 
@st.cache_data(ttl=3600)
def load_data(symbol, period):
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()
 
 
def add_indicators(df):
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    std = df["Close"].rolling(20).std()
    df["BB_up"] = df["MA20"] + 2 * std
    df["BB_low"] = df["MA20"] - 2 * std
    df["Return"] = df["Close"].pct_change()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    return df
 
 
def style(fig, height):
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    # hide weekends so there are no empty gaps in the chart
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig
 
 
data = load_data(ticker, period)
if data.empty:
    st.error("Data nahi mila. Symbol check karo (jaise TCS.NS, INFY.NS).")
    st.stop()
data = add_indicators(data)
 
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🤖 Model Comparison", "⚖️ Compare Stocks"])
 
# =====================================================
# TAB 1: Analysis
# =====================================================
with tab1:
    last = data["Close"].iloc[-1]
    change = data["Return"].iloc[-1] * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"₹{last:,.2f}", f"{change:.2f}%")
    c2.metric("Period High", f"₹{data['High'].max():,.2f}")
    c3.metric("Period Low", f"₹{data['Low'].min():,.2f}")
    c4.metric("Volatility (ann.)", f"{data['Return'].std() * np.sqrt(252) * 100:.1f}%")
 
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=(f"{ticker} Price", "Volume", "RSI (14)"),
    )
 
    # --- Price ---
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="Price",
            increasing_line_color=UP, decreasing_line_color=DOWN,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"], name="Close",
            line=dict(color="#2962ff", width=2),
            fill="tozeroy", fillcolor="rgba(41,98,255,0.08)",
        ), row=1, col=1)
 
    if show_ma:
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20",
                                 line=dict(color="#ffa726", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50",
                                 line=dict(color="#ab47bc", width=1.3)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_up"], name="BB upper",
                                 line=dict(color="rgba(160,160,160,0.6)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_low"], name="BB lower",
                                 line=dict(color="rgba(160,160,160,0.6)", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(160,160,160,0.08)"), row=1, col=1)
 
    # --- Volume ---
    vol_colors = np.where(data["Close"] >= data["Open"], UP, DOWN)
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.8), row=2, col=1)
 
    # --- RSI ---
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI",
                             line=dict(color="#b388ff", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=DOWN, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=UP, row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
 
    st.plotly_chart(style(fig, 780), width="stretch")
 
    rsi = data["RSI"].iloc[-1]
    st.caption(
        f"Current RSI: {rsi:.1f} → "
        + ("Overbought (>70)" if rsi > 70 else "Oversold (<30)" if rsi < 30 else "Neutral")
    )
 
    with st.expander("Raw data"):
        st.dataframe(data.tail(50))
 
# =====================================================
# TAB 2: Model comparison
# =====================================================
with tab2:
    st.subheader("Next-day Close Prediction: Linear Regression vs Random Forest")
 
    df = data[["Close", "MA50", "RSI"]].copy()
    df["Lag1"] = df["Close"].shift(1)
    df["Lag2"] = df["Close"].shift(2)
    df["Target"] = df["Close"].shift(-1)
    train_df = df.dropna()
    features = ["Close", "Lag1", "Lag2", "MA50", "RSI"]
 
    if len(train_df) < 60:
        st.warning("Is period mein data kam hai. Sidebar se 1y ya 2y chuno.")
    else:
        split = int(len(train_df) * 0.8)  # time-based split
        X_train, X_test = train_df[features][:split], train_df[features][split:]
        y_train, y_test = train_df["Target"][:split], train_df["Target"][split:]
 
        lr = LinearRegression().fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
        lr_pred, rf_pred = lr.predict(X_test), rf.predict(X_test)
        naive_pred = X_test["Close"].values  # "tomorrow = today"
 
        results = pd.DataFrame({
            "Model": ["Naive (today = tomorrow)", "Linear Regression", "Random Forest"],
            "MAE (₹)": [
                mean_absolute_error(y_test, naive_pred),
                mean_absolute_error(y_test, lr_pred),
                mean_absolute_error(y_test, rf_pred),
            ],
        }).sort_values("MAE (₹)")
        results["MAE (₹)"] = results["MAE (₹)"].round(2)
        st.dataframe(results, hide_index=True, width="stretch")
 
        latest = df[features].dropna().iloc[[-1]]
        last_close = data["Close"].iloc[-1]
        lr_next, rf_next = lr.predict(latest)[0], rf.predict(latest)[0]
        m1, m2 = st.columns(2)
        m1.metric("Linear Regression: next close", f"₹{lr_next:,.2f}", f"{(lr_next - last_close) / last_close * 100:.2f}%")
        m2.metric("Random Forest: next close", f"₹{rf_next:,.2f}", f"{(rf_next - last_close) / last_close * 100:.2f}%")
 
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name="Actual",
                                  line=dict(color="white", width=2)))
        figp.add_trace(go.Scatter(x=y_test.index, y=lr_pred, name="Linear Regression",
                                  line=dict(color="#42a5f5", width=1.5)))
        figp.add_trace(go.Scatter(x=y_test.index, y=rf_pred, name="Random Forest",
                                  line=dict(color="#ffa726", width=1.5)))
        figp.update_layout(title="Test period: Actual vs Predicted", yaxis_title="Price (₹)")
        st.plotly_chart(style(figp, 420), width="stretch")
 
        st.info(
            "Tip: Agar koi model 'Naive' se behtar nahi hai, to wo asli mein kuch naya "
            "nahi seekh raha. Stock prediction mushkil hota hai, isliye yeh normal hai."
        )
 
    st.warning("⚠️ Yeh sirf learning project hai. Isse investment advice mat samjhna.")
 
# =====================================================
# TAB 3: Compare two stocks
# =====================================================
with tab3:
    st.subheader(f"{ticker} vs {ticker2}")
    if not ticker2:
        st.info("Sidebar mein compare karne ke liye doosra symbol daalo.")
    else:
        data2 = load_data(ticker2, period)
        if data2.empty:
            st.error(f"{ticker2} ka data nahi mila.")
        else:
            both = pd.concat(
                [data["Close"].rename(ticker), data2["Close"].rename(ticker2)], axis=1
            ).dropna()
            normalized = both / both.iloc[0] * 100
 
            figc = go.Figure()
            for col, color in zip(normalized.columns, ["#42a5f5", "#ffa726"]):
                figc.add_trace(go.Scatter(x=normalized.index, y=normalized[col],
                                          name=col, line=dict(color=color, width=2)))
            figc.add_hline(y=100, line_dash="dash", line_color="gray")
            figc.update_layout(title="Growth of ₹100 invested", yaxis_title="Value (₹)")
            st.plotly_chart(style(figc, 450), width="stretch")
 
            total_ret = (both.iloc[-1] / both.iloc[0] - 1) * 100
            vol = both.pct_change().std() * np.sqrt(252) * 100
            summary = pd.DataFrame({
                "Total Return (%)": total_ret.round(2),
                "Volatility (ann. %)": vol.round(2),
            })
            st.dataframe(summary, width="stretch")
            st.caption(f"Correlation of daily returns: {both.pct_change().corr().iloc[0, 1]:.2f}")
 