"""
Zepto & Blinkit Chennai — Customer Segmentation Streamlit App
Run: streamlit run app.py
Requires: kmeans_model.pkl, scaler.pkl, pca_model.pkl,
          chennai_customer_segments.csv, cluster_metadata.json
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os

# ─── Base directory (folder where app.py lives) ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Removed duplicate BASE_DIR


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chennai Q-Commerce Segments",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main { background: #0D1117; }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00B4D8 0%, #8338EC 50%, #FF6B35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #8B949E;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #E6EDF3;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .segment-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #161B22;
        border-left: 4px solid #00B4D8;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #C9D1D9;
        font-size: 0.9rem;
    }
    .stSelectbox > div > div, .stSlider > div { color: #C9D1D9; }
    div[data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #E6EDF3;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #21262D;
    }
</style>
""", unsafe_allow_html=True)

# ─── Segment Config ───────────────────────────────────────────────────────────
SEGMENT_CONFIG = {
    0: {
        "name": "💎 Premium Loyalists",
        "color": "#FF6B35",
        "badge_bg": "#3D1F0F",
        "badge_color": "#FF6B35",
        "desc": "High-value, frequent, loyal customers with premium order values.",
        "strategy": "Exclusive membership perks, early access to new SKUs, concierge-style service, VIP flash sales.",
        "kpis": ["High AOV (₹700+)", "20+ orders/month", "80%+ reorder rate", "Low churn risk"],
    },
    1: {
        "name": "🧪 Occasional Browsers",
        "color": "#00B4D8",
        "badge_bg": "#0A2533",
        "badge_color": "#00B4D8",
        "desc": "Infrequent users still exploring the platform. High conversion potential.",
        "strategy": "Re-engagement push notifications, first-time category deals, 'try again' cashback offers.",
        "kpis": ["3–6 orders/month", "Low session frequency", "Trial phase", "High acquisition cost"],
    },
    2: {
        "name": "🔥 Bargain Hunters",
        "color": "#8338EC",
        "badge_bg": "#1E0F3D",
        "badge_color": "#8338EC",
        "desc": "Price-sensitive, promotion-driven buyers who respond strongly to discounts.",
        "strategy": "Flash sale targeting, bundle deals, BOGO offers, coupon personalization.",
        "kpis": ["60%+ discount usage", "Mid-range orders", "Price-sensitive", "Promotion-driven"],
    },
    3: {
        "name": "⭐ Regular Mainstream",
        "color": "#06D6A0",
        "badge_bg": "#0A2A20",
        "badge_color": "#06D6A0",
        "desc": "Steady, reliable mid-segment customers with good growth potential.",
        "strategy": "Loyalty program enrollment, subscription push, category expansion upsell.",
        "kpis": ["10–15 orders/month", "Balanced spend", "Good retention", "Upsell potential"],
    },
}

FEATURES = [
    'monthly_orders', 'avg_order_value', 'monthly_spend',
    'reorder_rate', 'discount_usage_pct', 'app_sessions_per_week',
    'num_categories_ordered', 'membership', 'tenure_months',
    'days_since_last_order', 'complaints_filed', 'ratings_given'
]

# ─── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        pca    = joblib.load(os.path.join(BASE_DIR, "pca_model.pkl"))
        return kmeans, scaler, pca, True
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None, None, None, False

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "chennai_customer_segments.csv"))
        
        # Validate required columns
        required_cols = ['segment', 'platform', 'locality', 'monthly_orders', 
                        'avg_order_value', 'monthly_spend', 'reorder_rate', 
                        'discount_usage_pct', 'app_sessions_per_week', 
                        'num_categories_ordered', 'membership', 'tenure_months', 
                        'days_since_last_order', 'complaints_filed', 'ratings_given']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.sidebar.error(f"Missing columns in CSV: {missing_cols}")
            return None, False
            
        return df, True
    except Exception as e:
        st.sidebar.error(f"Data load error: {e}")
        return None, False


@st.cache_data
def load_meta():
    try:
        with open(os.path.join(BASE_DIR, "cluster_metadata.json")) as f:
            return json.load(f), True
    except Exception:
        return {}, False

def show_file_debug():
    st.markdown("### 🔍 File Path Debug")
    st.code(f"App directory (BASE_DIR): {BASE_DIR}")
    expected_files = [
        "kmeans_model.pkl", "scaler.pkl", "pca_model.pkl",
        "chennai_customer_segments.csv", "cluster_metadata.json"
    ]
    rows = []
    for f in expected_files:
        full_path = os.path.join(BASE_DIR, f)
        exists = os.path.exists(full_path)
        rows.append({"File": f, "Status": "✅ Found" if exists else "❌ Missing", "Full Path": full_path})
    st.table(pd.DataFrame(rows))


kmeans, scaler, pca, models_ok = load_models()
df, data_ok = load_data()
meta, meta_ok = load_meta()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 Quick Commerce\n**Segmentation App**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Predict Segment", "📋 Data Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    **Data:** Zepto & Blinkit Chennai  
    **Model:** K-Means (K=4)  
    **Customers:** 2,000  
    **Features:** 12
    """)
    if models_ok:
        st.success("✅ Models Loaded")
    else:
        st.error("❌ Models not found")
        if st.button("🔍 Debug file paths"):
            st.session_state["show_debug"] = True
    if data_ok:
        st.success("✅ Dataset Loaded")
    else:
        st.error("❌ Dataset not found")

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">Zepto & Blinkit Chennai<br>Customer Segmentation</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">K-Means ML · 2,000 Customers · 4 Segments · Chennai Quick Commerce Intelligence</p>', unsafe_allow_html=True)

# ─── Debug panel ─────────────────────────────────────────────────────────────
if st.session_state.get("show_debug"):
    show_file_debug()
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    if not data_ok:
        st.error("Dataset not found. Please run the Colab notebook and place `chennai_customer_segments.csv` here.")
        st.stop()

    # ── Top KPI Row ────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Avg Monthly Orders", f"{df['monthly_orders'].mean():.1f}")
    with col3:
        st.metric("Avg Order Value", f"₹{df['avg_order_value'].mean():.0f}")
    with col4:
        st.metric("Avg Monthly Spend", f"₹{df['monthly_spend'].mean():.0f}")
    with col5:
        st.metric("Membership Rate", f"{df['membership'].mean()*100:.1f}%")

    st.markdown('<p class="section-header">Segment Overview</p>', unsafe_allow_html=True)

    # ── Segment Distribution + Spend ─────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        seg_counts = df['segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        colors_list = [SEGMENT_CONFIG[c]["color"] for c in range(4)]
        fig_pie = px.pie(
            seg_counts, names='Segment', values='Count',
            color_discrete_sequence=colors_list,
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#C9D1D9', legend_font_size=11,
            title=dict(text="Customer Distribution by Segment", font=dict(size=14, color='#E6EDF3')),
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        seg_spend = df.groupby('segment')['monthly_spend'].mean().reset_index()
        seg_spend.columns = ['Segment', 'Avg Monthly Spend (₹)']
        fig_bar = px.bar(
            seg_spend, x='Segment', y='Avg Monthly Spend (₹)',
            color='Segment', color_discrete_sequence=colors_list,
            text_auto='.0f'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#C9D1D9', showlegend=False,
            title=dict(text="Avg Monthly Spend by Segment", font=dict(size=14, color='#E6EDF3')),
            xaxis=dict(gridcolor='#21262D'), yaxis=dict(gridcolor='#21262D'),
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Platform Split by Segment ─────────────────────────────────────────────
    st.markdown('<p class="section-header">Platform & Behavioral Breakdown</p>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        plat_seg = df.groupby(['segment', 'platform']).size().reset_index(name='count')
        fig_plat = px.bar(
            plat_seg, x='segment', y='count', color='platform',
            barmode='group', color_discrete_sequence=['#00B4D8', '#FF6B35', '#8338EC'],
            title="Platform Preference by Segment"
        )
        fig_plat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#C9D1D9', xaxis=dict(gridcolor='#21262D'),
            yaxis=dict(gridcolor='#21262D'),
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_plat, use_container_width=True)

    with col_d:
        box_data = []
        for seg in df['segment'].unique():
            vals = df[df['segment'] == seg]['avg_order_value'].values
            box_data.append(go.Box(y=vals, name=seg, marker_color=colors_list[list(df['segment'].unique()).index(seg)]))
        fig_box = go.Figure(data=box_data)
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#C9D1D9', showlegend=False,
            title=dict(text="Avg Order Value Distribution", font=dict(size=14, color='#E6EDF3')),
            xaxis=dict(gridcolor='#21262D'), yaxis=dict(gridcolor='#21262D', title='₹'),
            margin=dict(t=50, b=60, l=0, r=0)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Locality Heatmap ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Locality Analysis</p>', unsafe_allow_html=True)

    loc_seg = df.groupby(['locality', 'segment']).size().unstack(fill_value=0)
    fig_heat = px.imshow(
        loc_seg.values,
        x=loc_seg.columns.tolist(),
        y=loc_seg.index.tolist(),
        color_continuous_scale='Blues',
        title="Customer Count: Locality × Segment",
        labels=dict(color="Customers")
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#C9D1D9', height=380,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Segment Cards ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Segment Strategy Cards</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (cid, cfg) in enumerate(SEGMENT_CONFIG.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.1rem;font-weight:700;color:{cfg['color']};margin-bottom:0.5rem">{cfg['name']}</div>
                <div style="color:#8B949E;font-size:0.83rem;margin-bottom:0.8rem">{cfg['desc']}</div>
                <div style="font-size:0.8rem;font-weight:600;color:#E6EDF3;margin-bottom:0.3rem">🎯 Strategy</div>
                <div style="color:#C9D1D9;font-size:0.8rem">{cfg['strategy']}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICT SEGMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Segment":

    st.markdown('<p class="section-header">🔮 Predict Customer Segment</p>', unsafe_allow_html=True)
    st.markdown("Enter a new customer's details below to predict which segment they belong to.")

    if not models_ok:
        st.error("Model files not found. Run the Colab notebook first and download the .pkl files.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📦 Order Behavior**")
        monthly_orders = st.slider("Monthly Orders", 1, 45, 12)
        avg_order_value = st.slider("Avg Order Value (₹)", 80, 1500, 400)
        monthly_spend = monthly_orders * avg_order_value
        st.info(f"💰 Est. Monthly Spend: **₹{monthly_spend:,.0f}**")
        reorder_rate = st.slider("Reorder Rate", 0.1, 1.0, 0.6, 0.01)
        num_categories = st.slider("Categories Ordered", 1, 6, 3)

    with col2:
        st.markdown("**📱 Engagement**")
        app_sessions = st.slider("App Sessions / Week", 1.0, 30.0, 7.0, 0.5)
        days_since_last = st.slider("Days Since Last Order", 0, 60, 3)
        ratings_given = st.slider("Ratings Given", 0, 30, 5)
        complaints = st.slider("Complaints Filed", 0, 5, 0)

    with col3:
        st.markdown("**👤 Profile**")
        discount_usage = st.slider("Discount Usage %", 0.0, 100.0, 30.0, 0.5)
        membership = st.selectbox("Has Membership?", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 1, 36, 12)
        platform = st.selectbox("Platform", ["Blinkit", "Zepto", "Both"])
        locality = st.selectbox("Locality", [
            'Anna Nagar', 'T Nagar', 'Adyar', 'Velachery', 'OMR',
            'Porur', 'Chromepet', 'Tambaram', 'Nungambakkam', 'Mylapore',
            'Sholinganallur', 'Perungudi', 'Guindy', 'Kodambakkam', 'Ambattur'
        ])

    st.markdown("---")
    if st.button("🔮 Predict My Segment", type="primary", use_container_width=True):

        mem_val = 1 if membership == "Yes" else 0
        input_data = np.array([[
            monthly_orders, avg_order_value, monthly_spend,
            reorder_rate, discount_usage, app_sessions,
            num_categories, mem_val, tenure,
            days_since_last, complaints, ratings_given
        ]])

        input_scaled = scaler.transform(input_data)
        cluster_id = kmeans.predict(input_scaled)[0]
        cfg = SEGMENT_CONFIG[cluster_id]

        # Distances to all centroids
        distances = kmeans.transform(input_scaled)[0]
        confidence = 1 - (distances[cluster_id] / distances.sum())

        st.markdown("---")
        result_col, detail_col = st.columns([1, 1])

        with result_col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{cfg['color']};border-width:2px">
                <div style="font-size:0.85rem;color:#8B949E;margin-bottom:0.3rem">PREDICTED SEGMENT</div>
                <div style="font-size:2rem;font-weight:700;color:{cfg['color']};margin-bottom:0.5rem">{cfg['name']}</div>
                <div style="color:#C9D1D9;font-size:0.9rem;margin-bottom:1rem">{cfg['desc']}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#E6EDF3">Confidence Score</div>
                <div style="font-size:1.5rem;font-family:'JetBrains Mono';color:{cfg['color']}">{confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**🎯 Recommended Strategy**")
            st.markdown(f'<div class="insight-box">{cfg["strategy"]}</div>', unsafe_allow_html=True)

        with detail_col:
            st.markdown("**📊 Distance to All Segments**")
            seg_names = [SEGMENT_CONFIG[i]['name'] for i in range(4)]
            seg_colors = [SEGMENT_CONFIG[i]['color'] for i in range(4)]
            fig_dist = go.Figure(go.Bar(
                x=seg_names, y=distances,
                marker_color=seg_colors,
                text=[f"{d:.2f}" for d in distances],
                textposition='auto'
            ))
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#C9D1D9', showlegend=False,
                yaxis_title="Distance (lower = closer)",
                xaxis=dict(gridcolor='#21262D'), yaxis=dict(gridcolor='#21262D'),
                margin=dict(t=20, b=60, l=0, r=0), height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("**✅ Key Characteristics**")
            for kpi in cfg['kpis']:
                st.markdown(f"- {kpi}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Data Explorer":

    st.markdown('<p class="section-header">📋 Data Explorer</p>', unsafe_allow_html=True)

    if not data_ok:
        st.error("Dataset not found. Run the Colab notebook first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        seg_filter = st.multiselect("Filter by Segment", df['segment'].unique().tolist(),
                                     default=df['segment'].unique().tolist())
    with col2:
        plat_filter = st.multiselect("Filter by Platform", df['platform'].unique().tolist(),
                                      default=df['platform'].unique().tolist())
    with col3:
        loc_filter = st.multiselect("Filter by Locality", df['locality'].unique().tolist(),
                                     default=df['locality'].unique().tolist())

    filtered = df[
        df['segment'].isin(seg_filter) &
        df['platform'].isin(plat_filter) &
        df['locality'].isin(loc_filter)
    ]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} customers**")

    # Scatter explorer
    col_x, col_y = st.columns(2)
    num_cols = ['monthly_orders', 'avg_order_value', 'monthly_spend',
                'reorder_rate', 'discount_usage_pct', 'app_sessions_per_week', 'tenure_months']
    with col_x:
        x_col = st.selectbox("X Axis", num_cols, index=0)
    with col_y:
        y_col = st.selectbox("Y Axis", num_cols, index=1)

    fig_scatter = px.scatter(
        filtered, x=x_col, y=y_col, color='segment',
        color_discrete_sequence=[cfg['color'] for cfg in SEGMENT_CONFIG.values()],
        hover_data=['customer_id', 'locality', 'platform'],
        opacity=0.65, title=f"{x_col} vs {y_col} by Segment"
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#C9D1D9', height=420,
        xaxis=dict(gridcolor='#21262D'), yaxis=dict(gridcolor='#21262D'),
        margin=dict(t=50, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Raw table
    st.markdown('<p class="section-header">Raw Data Table</p>', unsafe_allow_html=True)
    show_cols = ['customer_id', 'locality', 'platform', 'segment',
                 'monthly_orders', 'avg_order_value', 'monthly_spend',
                 'reorder_rate', 'membership', 'tenure_months']
    st.dataframe(
        filtered[show_cols].reset_index(drop=True),
        use_container_width=True, height=350
    )

    # Download
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data as CSV", csv,
                       "filtered_segments.csv", "text/csv")