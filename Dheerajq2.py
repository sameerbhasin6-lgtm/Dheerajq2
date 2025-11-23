import streamlit as st
import pandas as pd
import numpy as np
# We import these globally, but we will also import the solver locally to fix your NameError
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PriceAi | Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM AESTHETICS & CSS (THEME V3) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #334155;
        background-color: #fafafa;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: white;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border-bottom: 1px solid #f1f5f9;
    }
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1e293b; margin: 0; letter-spacing: -1px; }
    .sub-title { font-size: 1.1rem; color: #64748b; margin-top: 0.5rem; font-weight: 300; }

    /* KPI Box */
    .kpi-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        border-top: 4px solid #cbd5e1;
        text-align: left;
        transition: transform 0.2s;
    }
    .kpi-box:hover { transform: translateY(-3px); }
    .kpi-label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
    .kpi-val { font-size: 2rem; font-weight: 700; color: #0f172a; margin: 0.5rem 0; }
    .kpi-sub { font-size: 0.9rem; font-weight: 500; }
    
    /* Specific Colors */
    .border-violet { border-top-color: #8b5cf6 !important; }
    .border-rose { border-top-color: #f43f5e !important; }
    .border-emerald { border-top-color: #10b981 !important; }
    .border-amber { border-top-color: #f59e0b !important; }
    
    .text-violet { color: #8b5cf6; }
    .text-rose { color: #f43f5e; }
    .text-emerald { color: #10b981; }

    /* Pricing Section */
    .price-grid {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 3rem 0;
        align-items: center;
    }
    .p-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
        min-width: 160px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
    }
    .p-name { font-size: 0.85rem; font-weight: 600; color: #64748b; margin-bottom: 5px; text-transform: uppercase; }
    .p-cost { font-size: 1.4rem; font-weight: 700; color: #334155; }

    /* Hero Bundle */
    .hero-bundle {
        background: linear-gradient(135deg, #2e1065 0%, #7c3aed 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        transform: scale(1.1);
        box-shadow: 0 20px 40px -10px rgba(124, 58, 237, 0.4);
        position: relative;
        border: none;
        min-width: 220px;
        z-index: 10;
    }
    .hero-bundle .p-name { color: rgba(255,255,255,0.7); }
    .hero-bundle .p-cost { color: white; font-size: 2.2rem; }
    .hero-tag {
        position: absolute;
        top: -10px; left: 50%; transform: translateX(-50%);
        background: #f43f5e; color: white;
        padding: 4px 12px; font-size: 0.75rem; font-weight: 700;
        border-radius: 20px; text-transform: uppercase;
        box-shadow: 0 4px 10px rgba(244, 63, 94, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    FILE_NAME = "Samsung_Sankalp.csv"
    try:
        df = pd.read_csv(FILE_NAME)
        return df
    except FileNotFoundError:
        st.error(f"🔴 System Error: Data file '{FILE_NAME}' not found.")
        st.stop()
        
# --- 2. OPTIMIZATION ENGINE ---

def calculate_baseline(df, products):
    total_rev = 0
    for prod in products:
        wtp = df[prod].values
        candidates = np.unique(wtp)
        best_r = 0
        for p in candidates:
            r = p * np.sum(wtp >= p)
            if r > best_r: best_r = r
        total_rev += best_r
    return total_rev

@st.cache_data(show_spinner=False)
def solve_pricing(df, products):
    # FIX: Importing inside the function to prevent NameError in Streamlit Cloud
    from scipy.optimize import differential_evolution
    
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)

    def objective(prices):
        indiv_prices = np.array(prices[:n_prods])
        bundle_price = prices[n_prods]
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bundle_price
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        buy_indiv = (~buy_bundle) & (surplus_indiv > 0)
        rev_bundle = np.sum(buy_bundle) * bundle_price
        items_bought_mask = (wtp_matrix >= indiv_prices) & buy_indiv[:, None]
        rev_indiv = np.sum(items_bought_mask * indiv_prices)
        return -(rev_bundle + rev_indiv)

    bounds = []
    for i in range(n_prods):
        max_w = np.max(wtp_matrix[:, i])
        bounds.append((0, max_w * 1.5)) 
    bounds.append((0, np.max(bundle_sum_values)))

    # Running the solver with the local import
    res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=50, popsize=15, tol=0.01, seed=42)
    return res.x, -res.fun

def get_customer_breakdown(df, products, optimal_prices):
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = optimal_prices[:n_prods]
    bundle_price = optimal_prices[n_prods]
    
    rows = []
    for i in range(len(df)):
        s_indiv = np.sum(np.maximum(wtp_matrix[i] - indiv_prices, 0))
        s_bundle = bundle_sum_values[i] - bundle_price
        decision = "None"
        revenue = 0
        surplus = 0
        items = "-"
        if s_bundle >= s_indiv and s_bundle >= 0:
            decision = "Bundle"
            revenue = bundle_price
            surplus = s_bundle
            items = "All Items"
        elif s_indiv > 0:
            decision = "Individual"
            surplus = s_indiv
            bought_indices = np.where(wtp_matrix[i] >= indiv_prices)[0]
            items = ", ".join([products[k] for k in bought_indices])
            revenue = np.sum(indiv_prices[bought_indices])
            
        rows.append({
            "Customer ID": i + 1, "Decision": decision, 
            "Items Bought": items.replace("Samsung_", "").replace("_", " "),
            "Revenue": revenue, "Consumer Surplus": surplus
        })
    return pd.DataFrame(rows)

def generate_demand_curve(df, products, optimal_prices):
    wtp_matrix = df[products].values
    bundle_sum_values = df[products].sum(axis=1).values
    n_prods = len(products)
    indiv_prices = optimal_prices[:n_prods]
    max_val = np.max(bundle_sum_values)
    price_points = np.linspace(0, max_val, 100)
    demand = []
    for bp in price_points:
        surplus_indiv = np.sum(np.maximum(wtp_matrix - indiv_prices, 0), axis=1)
        surplus_bundle = bundle_sum_values - bp
        buy_bundle = (surplus_bundle >= surplus_indiv) & (surplus_bundle >= 0)
        demand.append(np.sum(buy_bundle))
    return pd.DataFrame({"Price": price_points, "Demand": demand})

# --- MAIN APP ---

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">PriceAi Analytics</h1>
        <div class="sub-title">Advanced Mixed-Bundling & Revenue Optimization</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    products = df.columns.tolist()
    
    with st.spinner("🔮 Analyzing Elasticity & Optimizing..."):
        baseline_rev = calculate_baseline(df, products)
        opt_prices, max_rev = solve_pricing(df, products)
        customer_df = get_customer_breakdown(df, products, opt_prices)
        
        # Core Metrics
        total_surplus = customer_df['Consumer Surplus'].sum()
        uplift = ((max_rev - baseline_rev) / baseline_rev) * 100
        bundle_price = opt_prices[-1]
        sum_indiv_opt = np.sum(opt_prices[:-1])
        discount = ((sum_indiv_opt - bundle_price) / sum_indiv_opt) * 100
        bundle_adoption = (len(customer_df[customer_df['Decision'] == 'Bundle']) / len(df)) * 100
        
        # --- 1. KPI RIBBON ---
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f'<div class="kpi-box border-violet"><div class="kpi-label">Forecasted Revenue</div><div class="kpi-val">₹{max_rev:,.0f}</div><div class="kpi-sub text-violet">▲ {uplift:.1f}% vs Baseline</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-box border-emerald"><div class="kpi-label">Bundle Adoption</div><div class="kpi-val">{bundle_adoption:.0f}%</div><div class="kpi-sub text-emerald">Conversion Rate</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi-box border-rose"><div class="kpi-label">Bundle Discount</div><div class="kpi-val">{discount:.1f}%</div><div class="kpi-sub text-rose">Consumer Savings</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="kpi-box border-amber"><div class="kpi-label">Consumer Surplus</div><div class="kpi-val">₹{total_surplus:,.0f}</div><div class="kpi-sub" style="color:#f59e0b">Value Retained</div></div>', unsafe_allow_html=True)

        # --- 2. PRICING PODIUM ---
        st.markdown('<div class="price-grid">', unsafe_allow_html=True)
        
        p_html = '<div class="price-grid">'
        # Individual Items
        for i, prod in enumerate(products):
            p_opt = opt_prices[i]
            clean_name = prod.replace("Samsung_", "").replace("_", " ")
            p_html += f'<div class="p-card"><div class="p-name">{clean_name}</div><div class="p-cost">₹{p_opt:,.0f}</div></div>'
            
        # Bundle (Hero)
        p_html += f'<div class="hero-bundle"><div class="hero-tag">Best Value</div><div class="p-name">All-In Bundle</div><div class="p-cost">₹{bundle_price:,.0f}</div></div>'
        p_html += '</div>'
        st.markdown(p_html, unsafe_allow_html=True)

        # --- 3. TABS ---
        tab1, tab2, tab3 = st.tabs(["📢 Strategic Insights", "👥 Customer Segments", "📈 Market Simulation"])
        
        # TAB 1: STRATEGY
        with tab1:
            col_strat, col_chart = st.columns([1.5, 1])
            with col_strat:
                st.markdown("#### 🧠 Pricing Logic")
                if discount > 15:
                    st.info("💡 **Strategy: Volume Driver.** The AI has identified that deep discounting triggers a massive intake of the bundle, outweighing the loss of margin on individual units.")
                else:
                    st.info("💡 **Strategy: Premium Extraction.** The AI has identified high brand loyalty. A small discount is sufficient to nudge users, maintaining high margins.")
                
                st.markdown(f"""
                <ul style="color: #475569; line-height: 1.8;">
                    <li><b>Anchoring Effect:</b> Individual items are priced high to make the <b>₹{bundle_price:,.0f}</b> bundle look like a 'steal'.</li>
                    <li><b>Cross-Selling:</b> You are effectively cross-selling items by offering a total saving of <b>₹{(sum_indiv_opt - bundle_price):,.0f}</b>.</li>
                </ul>
                """, unsafe_allow_html=True)
                
            with col_chart:
                rev_bundle = customer_df[customer_df['Decision'] == 'Bundle']['Revenue'].sum()
                rev_indiv = customer_df[customer_df['Decision'] == 'Individual']['Revenue'].sum()
                
                fig_pie = px.pie(
                    names=['Bundle Rev', 'Individual Rev'],
                    values=[rev_bundle, rev_indiv],
                    hole=0.5,
                    color_discrete_sequence=['#7c3aed', '#e2e8f0']
                )
                fig_pie.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)

        # TAB 2: DATA
        with tab2:
            st.markdown("#### Segment breakdown")
            st.dataframe(
                customer_df,
                column_config={
                    "Customer ID": st.column_config.NumberColumn(format="#%d"),
                    "Revenue": st.column_config.NumberColumn(format="₹%d"),
                    "Consumer Surplus": st.column_config.ProgressColumn(format="₹%d", max_value=int(customer_df['Consumer Surplus'].max())),
                },
                use_container_width=True,
                height=500,
                hide_index=True
            )

        # TAB 3: CHART
        with tab3:
            st.markdown("#### Demand Sensitivity Analysis")
            demand_data = generate_demand_curve(df, products, opt_prices)
            
            fig = px.area(demand_data, x="Price", y="Demand")
            fig.add_vline(x=bundle_price, line_dash="dash", line_color="#10b981", annotation_text="Optimal")
            
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Outfit, sans-serif", color="#64748b"),
                hovermode="x unified",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
            )
            fig.update_traces(line_color='#7c3aed', fillcolor='rgba(124, 58, 237, 0.1)')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
