import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

st.set_page_config(page_title="Olist AI Logistics", page_icon="🗺️", layout="wide")

COORDS = {
    "SP": [-23.5505, -46.6333], "RJ": [-22.9068, -43.1729],
    "MG": [-19.9167, -43.9345], "RS": [-30.0346, -51.2177],
    "PR": [-25.4284, -49.2733], "BA": [-12.9714, -38.5014],
    "CE": [-3.7172, -38.5431]
}

@st.cache_resource
def load_assets():
    models = joblib.load("shipping_models_advanced.joblib")
    with open("penalty.json", "r") as f: 
        penalty = json.load(f)
    with open("carriers.json", "r") as f: 
        carriers = json.load(f)
    return models, penalty["penalty_cost_brl"], carriers

models, BRL_PER_LATE, carriers = load_assets()

st.title("🗺️ Olist Smart Logistics")

tab1, tab2 = st.tabs(["Smart Shipping Engine", "Network Optimization (Hubs)"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📦 Live Shipment Data")
        seller_state = st.selectbox("Seller State", ["SP", "RJ", "MG", "RS", "PR"])
        customer_state = st.selectbox("Customer State", ["RJ", "SP", "MG", "BA", "CE"])
        
        display_category = st.selectbox("Product Category", [
            "Electronics & Tech Accessories", 
            "Home & Living", 
            "Beauty & Personal Care", 
            "Sports & Leisure"
        ])
        
        category_mapping = {
            "Electronics & Tech Accessories": "informatica_acessorios",
            "Home & Living": "cama_mesa_banho",
            "Beauty & Personal Care": "beleza_saude",
            "Sports & Leisure": "esporte_lazer"
        }
        category = category_mapping[display_category]
        
        base_freight = st.number_input("Base Freight Cost (BRL)", 10.0, 200.0, 25.0)
        
        st.markdown("---")
        st.subheader("External Factors (Business Logic)")
        
        weather = st.selectbox("Route Weather Condition ⛈️", ["Clear ☀", "Heavy Rain 🌧️", "Severe Storms 🌪️"])
        is_peak = st.checkbox("High Network Load (Peak Season)")
        
        run_engine = st.button("Run Smart Engine", use_container_width=True)

    with col2:
        st.subheader("📍 Multi-leg Routing Tracking")
        
        # استخدام خريطة ستريمليت الأصلية الثابتة والصاروخية
        start_lat, start_lon = COORDS[seller_state]
        end_lat, end_lon = COORDS[customer_state]
        
        map_df = pd.DataFrame({
            'lat': [start_lat, end_lat],
            'lon': [start_lon, end_lon],
            'color': ['#00FF00', '#FF0000']
        })
        st.map(map_df, latitude='lat', longitude='lon', size=50, color='color')

    if run_engine:
        order_data = {
            "seller_state": seller_state, "customer_state": customer_state,
            "lane": f"{seller_state}->{customer_state}", "seller_city": "unknown", "customer_city": "unknown",
            "product_category_name": category, "price": 150.0, "freight_value": base_freight, 
            "freight_ratio": base_freight / 150.0, "seller_late_rate": 0.1, "budget_days": 12.0,
            "carrier_month": 10, "carrier_dow": 2, "is_black_friday": 1 if is_peak else 0,
            "is_holiday_season": 0, "same_state": 1 if seller_state == customer_state else 0,
            "weight_g": 1200, "volume_cm3": 8000, "distance_km": 400.0
        }
        
        df_order = pd.DataFrame([order_data])
        for col in models["categories"]:
            if col in df_order.columns: 
                df_order[col] = df_order[col].astype("category")
                
        base_days = models["model_p50"].predict(df_order[models["features"]])[0]
        base_late_prob = models["model_late"].predict_proba(df_order[models["features"]])[0, 1]
        
        results = []
        for carrier_id, profile in carriers.items():
            carrier_days = base_days * profile["speed_multiplier"]
            carrier_cost = base_freight * profile["cost_multiplier"]
            
            penalty_multiplier = 1.0
            
            if is_peak and profile["type"] == "Local": 
                penalty_multiplier *= 1.5
            
            if weather == "Heavy Rain 🌧️":
                if profile["type"] != "Express": 
                    carrier_days *= 1.3  
                penalty_multiplier *= 1.2
            elif weather == "Severe Storms 🌪️":
                if profile["type"] != "Express": 
                    carrier_days *= 1.8  
                penalty_multiplier *= 1.5
                
            carrier_late_prob = min(base_late_prob * profile["late_prob_multiplier"] * penalty_multiplier, 1.0)
            expected_penalty = carrier_late_prob * BRL_PER_LATE
            
            results.append({
                "Carrier": profile["name"], 
                "Type": profile["type"],
                "Est. Days ⏱️": round(carrier_days, 1),
                "Cost (BRL) 💰": round(carrier_cost, 2), 
                "Late Risk ⚠️": f"{round(carrier_late_prob * 100, 1)}%",
                "Penalty Risk 💸": round(expected_penalty, 2),
                "Total Cost ⭐": round(carrier_cost + expected_penalty, 2)
            })
            
        st.markdown("---")
        st.subheader("Final Recommendations (Weather & Capacity Applied)")
        st.dataframe(pd.DataFrame(results).sort_values("Total Cost ⭐"), use_container_width=True, hide_index=True)


with tab2:
    st.subheader("📍 Customer Concentration & High Freight Cost Map")
    st.markdown("The current Olist system ships most products from São Paulo (SP). Red hotspots represent customers paying exceptionally high shipping costs. **Opening a Cross-Docking Hub in these areas will save millions in freight costs.**")
    
    # خريطة نقاط انتشار العملاء الثابتة والاحترافية
    hub_df = pd.DataFrame({
        'lat': [-12.9714, -3.7172, -22.9068, -23.5505, -19.9167],
        'lon': [-38.5014, -38.5431, -43.1729, -46.6333, -43.9345]
    })
    st.map(hub_df, latitude='lat', longitude='lon', size=100)
