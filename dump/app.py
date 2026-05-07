import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, os, warnings
warnings.filterwarnings('ignore')
 
st.set_page_config(
    page_title="PricePulse — Dynamic Ride Pricing",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
 
html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }
.main { background: #0d1117; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }
 
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0f2536 50%, #0d1117 100%);
    border: 1px solid #1e90ff25;
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 50%, #1e90ff0d 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #e2e8f0 0%, #1e90ff 50%, #00d4aa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.hero-sub { color: #64748b; font-size: 0.95rem; margin-top: 0.4rem; letter-spacing: 0.04em; }
.hero-badges { margin-top: 1rem; }
.badge {
    display: inline-block;
    background: #1e90ff15; border: 1px solid #1e90ff30;
    border-radius: 20px; padding: 0.2rem 0.8rem;
    font-size: 0.78rem; color: #7cb9ff; margin-right: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}
 
.kpi {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px; padding: 1.3rem 1.5rem;
    text-align: center; transition: border-color 0.2s;
}
.kpi:hover { border-color: #1e90ff55; }
.kpi-val { font-size: 1.85rem; font-weight: 700; color: #00d4aa; font-family: 'JetBrains Mono', monospace; }
.kpi-label { font-size: 0.75rem; color: #4b5563; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }
 
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #cbd5e1;
    border-left: 3px solid #1e90ff;
    padding-left: 0.7rem; margin: 1.4rem 0 0.9rem;
}
 
.pred-card {
    background: linear-gradient(145deg, #0d2137, #072b24);
    border: 2px solid #00d4aa55;
    border-radius: 18px; padding: 2rem; text-align: center;
}
.pred-val { font-size: 3.2rem; font-weight: 800; color: #00d4aa; font-family: 'JetBrains Mono', monospace; }
.pred-lbl { color: #64748b; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.pred-delta { font-size: 1rem; font-weight: 600; margin-top: 0.5rem; }
 
.chip {
    display: inline-block;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 999px; padding: 0.22rem 0.75rem;
    font-size: 0.78rem; color: #94a3b8; margin: 0.2rem;
}
 
.model-badge {
    background: linear-gradient(90deg,#1e90ff20,#00d4aa20);
    border: 1px solid #1e90ff40; border-radius: 10px;
    padding: 0.8rem 1.2rem; font-size: 0.9rem;
    color: #93c5fd; font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem; display: inline-block;
}
 
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
footer, header, #MainMenu { visibility: hidden; }
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: #1e2937 !important; color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)
 
# ── Artifacts ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
 
@st.cache_resource
def load_all():
    model   = joblib.load(f'{BASE}/best_model.pkl')
    scaler  = joblib.load(f'{BASE}/scaler.pkl')
    le_loc  = joblib.load(f'{BASE}/le_loc.pkl')
    le_loy  = joblib.load(f'{BASE}/le_loy.pkl')
    le_time = joblib.load(f'{BASE}/le_time.pkl')
    le_veh  = joblib.load(f'{BASE}/le_veh.pkl')
    return model, scaler, le_loc, le_loy, le_time, le_veh
 
@st.cache_data
def load_data():
    df = pd.read_csv(f'{BASE}/pricepulse_processed.csv')
    return df
 
@st.cache_data
def load_metrics():
    return pd.read_csv(f'{BASE}/model_metrics.csv', index_col=0)
 
model, scaler, le_loc, le_loy, le_time, le_veh = load_all()
df   = load_data()
mdf  = load_metrics()
best_name = open(f'{BASE}/best_model_name.txt').read().strip()
FEATURES  = open(f'{BASE}/features.txt').read().strip().split('\n')
 
# ── Predict helper ─────────────────────────────────────────────────────────────
def predict(riders, drivers, location, loyalty, past_rides, rating,
            time_of_day, vehicle, duration):
    loc_e  = int(le_loc.transform([location])[0])
    loy_e  = int(le_loy.transform([loyalty])[0])
    tim_e  = int(le_time.transform([time_of_day])[0])
    veh_e  = int(le_veh.transform([vehicle])[0])
    dsr    = riders / (drivers + 1)
    hd     = int(riders > df['Number_of_Riders'].quantile(0.75))
    ls     = int(drivers < df['Number_of_Drivers'].quantile(0.25))
    surge  = int(hd == 1 and ls == 1)
    diff   = riders - drivers
    prem   = int(vehicle == 'Premium')
    night  = int(time_of_day == 'Night')
    gold   = int(loyalty == 'Gold')
    X = np.array([[riders, drivers, loc_e, loy_e, past_rides, rating,
                   tim_e, veh_e, duration, dsr, hd, ls, surge, diff,
                   prem, night, gold]])
    X_sc = scaler.transform(X)
    return max(float(model.predict(X_sc)[0]), 0)
 
# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🚖 PricePulse</div>
  <div class="hero-sub">REAL-TIME DYNAMIC RIDE PRICING ENGINE &nbsp;·&nbsp; KAGGLE DATASET &nbsp;·&nbsp; ML-POWERED</div>
  <div class="hero-badges">
    <span class="badge">1,000 rides</span>
    <span class="badge">17 features</span>
    <span class="badge">7 models compared</span>
    <span class="badge">Lasso · RF · XGBoost</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ── TABS ───────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["🔮 Predict Fare", "📊 EDA Dashboard", "🤖 Model Arena", "📦 Batch Predict"])
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    left, right = st.columns([1.1, 1], gap="large")
 
    with left:
        st.markdown('<div class="section-title">Ride Parameters</div>', unsafe_allow_html=True)
 
        c1, c2 = st.columns(2)
        with c1:
            vehicle   = st.selectbox("Vehicle Type", ['Economy', 'Premium'])
            location  = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
            loyalty   = st.selectbox("Customer Loyalty", ['Regular', 'Silver', 'Gold'])
        with c2:
            time_of_day = st.selectbox("Time of Booking", ['Morning', 'Afternoon', 'Evening', 'Night'])
            duration    = st.slider("Expected Duration (min)", 10, 180, 60)
            past_rides  = st.slider("Past Rides", 0, 100, 30)
 
        st.markdown('<div class="section-title">Market Signals</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            riders  = st.slider("Active Riders", 20, 100, 65)
            rating  = st.slider("Avg Customer Rating", 3.5, 5.0, 4.3, 0.1)
        with c4:
            drivers = st.slider("Available Drivers", 5, 90, 20)
 
        dsr = riders / (drivers + 1)
        surge = (riders > df['Number_of_Riders'].quantile(0.75)) and (drivers < df['Number_of_Drivers'].quantile(0.25))
 
        # Live market indicators
        st.markdown('<div class="section-title">Live Market Indicators</div>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Demand/Supply Ratio", f"{dsr:.2f}", delta=f"{'+HIGH' if dsr>3 else 'Normal'}")
        mc2.metric("Surge Active", "⚡ YES" if surge else "✅ No")
        mc3.metric("Demand Pressure", "🔥 High" if riders > 75 else "🟢 Normal")
 
    with right:
        st.markdown('<div class="section-title">Fare Prediction</div>', unsafe_allow_html=True)
 
        pred_price = predict(riders, drivers, location, loyalty, past_rides,
                             rating, time_of_day, vehicle, duration)
 
        avg_cost = df['Historical_Cost_of_Ride'].mean()
        delta    = pred_price - avg_cost
        delta_pct = delta / avg_cost * 100
        clr = "#00d4aa" if delta >= 0 else "#ff6b6b"
        arrow = "▲" if delta >= 0 else "▼"
 
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-lbl">Recommended Fare</div>
            <div class="pred-val">₹ {pred_price:,.2f}</div>
            <div class="pred-delta" style="color:{clr}">
                {arrow} ₹{abs(delta):.2f} ({abs(delta_pct):.1f}%) vs avg fare
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_price,
            number={'prefix': '₹', 'valueformat': ',.1f', 'font': {'color': '#e2e8f0', 'size': 28}},
            title={'text': "Fare vs Historical Range", 'font': {'size': 12, 'color': '#64748b'}},
            gauge={
                'axis': {'range': [df['Historical_Cost_of_Ride'].min(),
                                   df['Historical_Cost_of_Ride'].max()],
                         'tickprefix': '₹', 'tickcolor': '#475569'},
                'bar': {'color': '#00d4aa', 'thickness': 0.25},
                'bgcolor': '#161b22', 'bordercolor': '#21262d',
                'steps': [
                    {'range': [df['Historical_Cost_of_Ride'].min(),
                               df['Historical_Cost_of_Ride'].quantile(0.33)], 'color': '#0f2918'},
                    {'range': [df['Historical_Cost_of_Ride'].quantile(0.33),
                               df['Historical_Cost_of_Ride'].quantile(0.66)], 'color': '#1a2f1a'},
                    {'range': [df['Historical_Cost_of_Ride'].quantile(0.66),
                               df['Historical_Cost_of_Ride'].max()], 'color': '#2d1f0a'},
                ],
                'threshold': {'line': {'color': '#1e90ff', 'width': 2},
                              'value': avg_cost}
            }
        ))
        fig_g.update_layout(height=230, paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=30, b=5, l=20, r=20))
        st.plotly_chart(fig_g, use_container_width=True)
 
        # Insights
        st.markdown("**💡 Pricing Signals**")
        chips = []
        if surge:           chips.append("⚡ Surge pricing active")
        if vehicle=='Premium': chips.append("💎 Premium vehicle markup")
        if loyalty=='Gold': chips.append("🥇 Gold loyalty discount")
        if time_of_day=='Night': chips.append("🌙 Night-time premium")
        if location=='Urban': chips.append("🏙️ Urban zone pricing")
        if dsr > 3:         chips.append("🔥 High demand-supply gap")
        if drivers < 15:    chips.append("⚠️ Low driver availability")
        if not chips:       chips.append("✅ Standard pricing conditions")
        for c in chips:
            st.markdown(f'<span class="chip">{c}</span>', unsafe_allow_html=True)
 
        # Sensitivity: riders vs fare
        st.markdown("**📉 Rider Count Sensitivity**")
        r_range = np.arange(20, 101, 5)
        p_range = [predict(r, drivers, location, loyalty, past_rides,
                           rating, time_of_day, vehicle, duration) for r in r_range]
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=r_range, y=p_range, mode='lines+markers',
                                   line=dict(color='#1e90ff', width=2.5),
                                   marker=dict(size=5), fill='tozeroy',
                                   fillcolor='rgba(30,144,255,0.08)'))
        fig_s.add_vline(x=riders, line_dash='dash', line_color='#00d4aa',
                        annotation_text=f"Now: {riders}", annotation_font_color='#00d4aa')
        fig_s.update_layout(height=170, paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=5, b=30, l=45, r=10),
                            xaxis=dict(title='Riders', color='#475569', gridcolor='#21262d'),
                            yaxis=dict(title='₹ Fare', color='#475569', gridcolor='#21262d'),
                            font_color='#94a3b8')
        st.plotly_chart(fig_s, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
 
    k1,k2,k3,k4,k5 = st.columns(5)
    for col, lbl, val in zip([k1,k2,k3,k4,k5], [
        "Total Rides","Avg Fare","Max Fare","Vehicle Types","Loyalty Tiers"],[
        f"{len(df):,}", f"₹{df['Historical_Cost_of_Ride'].mean():.0f}",
        f"₹{df['Historical_Cost_of_Ride'].max():.0f}",
        str(df['Vehicle_Type'].nunique()), str(df['Customer_Loyalty_Status'].nunique())
    ]):
        col.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)
 
    st.markdown("")
    r1, r2 = st.columns(2)
 
    with r1:
        st.markdown('<div class="section-title">Fare Distribution by Vehicle Type</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Historical_Cost_of_Ride', color='Vehicle_Type',
                           nbins=50, barmode='overlay', template='plotly_dark', opacity=0.75,
                           color_discrete_map={'Economy':'#1e90ff','Premium':'#00d4aa'})
        fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
 
    with r2:
        st.markdown('<div class="section-title">Avg Fare by Location & Time</div>', unsafe_allow_html=True)
        pivot = df.pivot_table('Historical_Cost_of_Ride', 'Time_of_Booking', 'Location_Category', aggfunc='mean')
        fig = px.imshow(pivot, text_auto='.0f', template='plotly_dark',
                        color_continuous_scale='Blues', aspect='auto')
        fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
 
    r2a, r2b = st.columns(2)
    with r2a:
        st.markdown('<div class="section-title">Demand vs Fare (Demand-Supply Ratio)</div>', unsafe_allow_html=True)
        df['demand_supply_ratio'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1)
        fig = px.scatter(df, x='demand_supply_ratio', y='Historical_Cost_of_Ride',
                         color='Vehicle_Type', template='plotly_dark', opacity=0.6,
                         trendline='ols', size_max=6,
                         color_discrete_map={'Economy':'#1e90ff','Premium':'#ff8c42'})
        fig.update_layout(height=330, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
 
    with r2b:
        st.markdown('<div class="section-title">Ride Duration vs Fare</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='Expected_Ride_Duration', y='Historical_Cost_of_Ride',
                         color='Location_Category', template='plotly_dark', opacity=0.6,
                         trendline='ols', size_max=6,
                         color_discrete_map={'Urban':'#1e90ff','Suburban':'#00d4aa','Rural':'#ff8c42'})
        fig.update_layout(height=330, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
 
    r3a, r3b = st.columns(2)
    with r3a:
        st.markdown('<div class="section-title">Fare by Loyalty Tier & Vehicle</div>', unsafe_allow_html=True)
        grp = df.groupby(['Customer_Loyalty_Status','Vehicle_Type'])['Historical_Cost_of_Ride'].mean().reset_index()
        fig = px.bar(grp, x='Customer_Loyalty_Status', y='Historical_Cost_of_Ride',
                     color='Vehicle_Type', barmode='group', template='plotly_dark',
                     color_discrete_map={'Economy':'#1e90ff','Premium':'#00d4aa'},
                     category_orders={'Customer_Loyalty_Status':['Regular','Silver','Gold']})
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
 
    with r3b:
        st.markdown('<div class="section-title">Fare Boxplot by Time of Booking</div>', unsafe_allow_html=True)
        fig = px.box(df, x='Time_of_Booking', y='Historical_Cost_of_Ride',
                     color='Time_of_Booking', template='plotly_dark',
                     category_orders={'Time_of_Booking':['Morning','Afternoon','Evening','Night']},
                     color_discrete_sequence=['#1e90ff','#00d4aa','#ff8c42','#c084fc'])
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
 
    st.markdown('<div class="section-title">Full Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', template='plotly_dark',
                    color_continuous_scale='RdBu_r', aspect='auto', zmin=-1, zmax=1)
    fig.update_layout(height=520, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL ARENA
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown(f'<div class="model-badge">🏆 Best Model: {best_name} &nbsp;|&nbsp; R² = {mdf.loc[best_name,"R2"]:.4f} &nbsp;|&nbsp; MAE = {mdf.loc[best_name,"MAE"]:.2f}</div>', unsafe_allow_html=True)
 
    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    best = mdf.loc[best_name]
    for col, lbl, val in zip([k1,k2,k3,k4],
        ["R² Score","MAE","RMSE","MAPE"],
        [f"{best['R2']:.4f}", f"{best['MAE']:.2f}", f"{best['RMSE']:.2f}", f"{best['MAPE']:.2f}%"]):
        col.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)
 
    st.markdown("")
 
    # Metrics table
    st.markdown('<div class="section-title">All Models Performance</div>', unsafe_allow_html=True)
    display = mdf[['R2','MAE','RMSE','MAPE','CV_R2']].copy()
    display = display.sort_values('R2', ascending=False)
    display.columns = ['R²','MAE','RMSE','MAPE (%)','CV R²']
    st.dataframe(display.style
                 .format({'R²':'{:.4f}','MAE':'{:.2f}','RMSE':'{:.2f}','MAPE (%)':'{:.2f}','CV R²':'{:.4f}'})
                 .background_gradient(subset=['R²'], cmap='Greens')
                 .background_gradient(subset=['MAE','RMSE'], cmap='Reds_r'),
                 use_container_width=True)
 
    ra, rb = st.columns(2)
    with ra:
        st.markdown('<div class="section-title">R² Score Comparison</div>', unsafe_allow_html=True)
        r2s = mdf['R2'].sort_values()
        colors = ['#00d4aa' if n==best_name else '#1e90ff' for n in r2s.index]
        fig = go.Figure(go.Bar(x=r2s.values, y=r2s.index, orientation='h',
                               marker_color=colors, text=[f'{v:.4f}' for v in r2s.values],
                               textposition='outside', textfont_color='#94a3b8'))
        fig.update_layout(height=320, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10),
                          xaxis_title='R²', yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
 
    with rb:
        st.markdown('<div class="section-title">MAE Comparison</div>', unsafe_allow_html=True)
        maes = mdf['MAE'].sort_values(ascending=False)
        colors_m = ['#00d4aa' if n==best_name else '#ff6b6b' for n in maes.index]
        fig = go.Figure(go.Bar(x=maes.values, y=maes.index, orientation='h',
                               marker_color=colors_m, text=[f'{v:.2f}' for v in maes.values],
                               textposition='outside', textfont_color='#94a3b8'))
        fig.update_layout(height=320, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10),
                          xaxis_title='MAE', yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
 
    st.markdown('<div class="section-title">Actual vs Predicted (Best 3 Models)</div>', unsafe_allow_html=True)
    y_test_arr = np.load(f'{BASE}/y_test.npy')
    y_pred_arr = np.load(f'{BASE}/best_y_pred.npy')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test_arr, y=y_pred_arr, mode='markers',
                             marker=dict(color='#1e90ff', opacity=0.55, size=7),
                             name=best_name))
    mn, mx = y_test_arr.min(), y_test_arr.max()
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                             line=dict(color='#ff6b6b', dash='dash', width=2), name='Perfect Fit'))
    fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10),
                      xaxis_title='Actual Fare (₹)', yaxis_title='Predicted Fare (₹)')
    st.plotly_chart(fig, use_container_width=True)
 
    st.markdown('<div class="section-title">Feature Coefficients — Lasso Regression</div>', unsafe_allow_html=True)
    if hasattr(model, 'coef_'):
        coef = pd.Series(model.coef_, index=FEATURES).sort_values()
        colors_c = ['#ff6b6b' if v < 0 else '#00d4aa' for v in coef.values]
        fig = go.Figure(go.Bar(x=coef.values, y=coef.index, orientation='h',
                               marker_color=colors_c,
                               text=[f'{v:.3f}' for v in coef.values],
                               textposition='outside', textfont_color='#94a3b8'))
        fig.update_layout(height=480, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(22,27,34,0.9)', margin=dict(t=10),
                          xaxis_title='Coefficient', yaxis_title='',
                          shapes=[dict(type='line', x0=0, x1=0, y0=-0.5, y1=len(coef)-0.5,
                                       line=dict(color='white', width=1, dash='dot'))])
        st.plotly_chart(fig, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown('<div class="section-title">Batch Price Prediction</div>', unsafe_allow_html=True)
    st.markdown("Generate predictions on a sample slice of the dataset or upload your own CSV.")
 
    c1, c2 = st.columns([1, 2])
    with c1:
        n_sample = st.slider("Sample size", 20, 200, 50, 10)
        run_batch = st.button("▶ Run Batch Prediction", type="primary")
    with c2:
        uploaded = st.file_uploader("Or upload a CSV", type='csv')
 
    batch = None
    if run_batch:
        batch = df.sample(n_sample, random_state=7).reset_index(drop=True)
    elif uploaded:
        batch = pd.read_csv(uploaded)
 
    if batch is not None:
        try:
            b = batch.copy()
            if 'demand_supply_ratio' not in b.columns:
                b['demand_supply_ratio'] = b['Number_of_Riders'] / (b['Number_of_Drivers'] + 1)
            b['high_demand_flag']  = (b['Number_of_Riders'] > df['Number_of_Riders'].quantile(0.75)).astype(int)
            b['low_supply_flag']   = (b['Number_of_Drivers'] < df['Number_of_Drivers'].quantile(0.25)).astype(int)
            b['surge_indicator']   = ((b['high_demand_flag']==1) & (b['low_supply_flag']==1)).astype(int)
            b['rider_driver_diff'] = b['Number_of_Riders'] - b['Number_of_Drivers']
            b['is_premium']   = (b['Vehicle_Type'] == 'Premium').astype(int)
            b['is_night']     = (b['Time_of_Booking'] == 'Night').astype(int)
            b['is_loyal_gold']= (b['Customer_Loyalty_Status'] == 'Gold').astype(int)
            b['Location_enc'] = le_loc.transform(b['Location_Category'].astype(str))
            b['Loyalty_enc']  = le_loy.transform(b['Customer_Loyalty_Status'].astype(str))
            b['Time_enc']     = le_time.transform(b['Time_of_Booking'].astype(str))
            b['Vehicle_enc']  = le_veh.transform(b['Vehicle_Type'].astype(str))
 
            X_b = scaler.transform(b[FEATURES].values)
            b['Predicted_Fare'] = np.maximum(model.predict(X_b), 0).round(2)
            if 'Historical_Cost_of_Ride' in b.columns:
                b['Actual_Fare'] = b['Historical_Cost_of_Ride'].round(2)
                b['Error']       = (b['Predicted_Fare'] - b['Actual_Fare']).round(2)
 
            show_cols = ['Vehicle_Type','Location_Category','Customer_Loyalty_Status',
                         'Time_of_Booking','Number_of_Riders','Number_of_Drivers',
                         'Expected_Ride_Duration','Predicted_Fare']
            if 'Actual_Fare' in b.columns: show_cols += ['Actual_Fare','Error']
            st.dataframe(b[show_cols].style.background_gradient(subset=['Predicted_Fare'], cmap='Blues'),
                         use_container_width=True)
 
            rb1, rb2 = st.columns(2)
            with rb1:
                fig = px.box(b, x='Vehicle_Type', y='Predicted_Fare', color='Vehicle_Type',
                             template='plotly_dark',
                             color_discrete_map={'Economy':'#1e90ff','Premium':'#00d4aa'})
                fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(22,27,34,0.9)', title='Predicted Fare by Vehicle',
                                  showlegend=False, margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True)
 
            with rb2:
                if 'Error' in b.columns:
                    fig = px.histogram(b, x='Error', nbins=25, template='plotly_dark',
                                       color_discrete_sequence=['#c084fc'])
                    fig.add_vline(x=0, line_dash='dash', line_color='white')
                    fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(22,27,34,0.9)', title='Prediction Error Distribution',
                                      margin=dict(t=40))
                    st.plotly_chart(fig, use_container_width=True)
 
            st.download_button("⬇ Download Predictions CSV",
                               b[show_cols].to_csv(index=False),
                               "pricepulse_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Batch prediction error: {e}")
 
st.markdown("---")
st.markdown('<center style="color:#374151;font-size:0.78rem;">PricePulse &nbsp;·&nbsp; Kaggle Dynamic Pricing Dataset &nbsp;·&nbsp; Lasso · Random Forest · XGBoost</center>', unsafe_allow_html=True)