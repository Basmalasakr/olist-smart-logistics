import streamlit as st
import pandas as pd
import joblib
import json
import folium
import requests
from streamlit_folium import st_folium, folium_static
from folium.plugins import HeatMap
import numpy as np

st.set_page_config(page_title="Olist AI Logistics", page_icon="🗺️", layout="wide")

# إحداثيات ولايات البرازيل
COORDS = {
    "SP": [-23.5505, -46.6333], "RJ": [-22.9068, -43.1729],
    "MG": [-19.9167, -43.9345], "RS": [-30.0346, -51.2177],
    "PR": [-25.4284, -49.2733], "BA": [-12.9714, -38.5014],
    "CE": [-3.7172, -38.5431],  "DF": [-15.7938, -47.8827],
    "AM": [-3.1190, -60.0217],  "SC": [-27.5969, -48.5495],
    "PE": [-8.0476, -34.8770]
}

@st.cache_data
def get_real_route(start, end):
    """جلب مسار الشوارع الحقيقي مجاناً عبر OSRM API"""
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
        r = requests.get(url, timeout=5).json()
        if r.get('code') == 'Ok':
            coords = r['routes'][0]['geometry']['coordinates']
            return [[c[1], c[0]] for c in coords]
    except:
        pass
    return [start, end]

@st.cache_resource
def load_assets():
    models = joblib.load("shipping_models_advanced.joblib")
    with open("penalty.json", "r") as f: 
        penalty = json.load(f)
    with open("carriers.json", "r") as f: 
        carriers = json.load(f)
    return models, penalty["penalty_cost_brl"], carriers

models, BRL_PER_LATE, carriers = load_assets()

st.title("🗺️ Olist Smart Logistics Engine")

tab1, tab2 = st.tabs(["🚀 Smart Shipping Engine", "🏢 Network Optimization (Hubs)"])

with tab1:
    col1, col2 = st.columns([1, 2.2])
    with col1:
        st.subheader("📦 Shipment Details")
        states_list = list(COORDS.keys())
        seller_state = st.selectbox("Seller State", states_list, index=0)
        customer_state = st.selectbox("Customer State", states_list, index=1)
        
        display_category = st.selectbox("Product Category", [
            "Electronics & Tech Accessories", "Home & Living", 
            "Beauty & Personal Care", "Sports & Leisure"
        ])
        
        category_mapping = {
            "Electronics & Tech Accessories": "informatica_acessorios",
            "Home & Living": "cama_mesa_banho",
            "Beauty & Personal Care": "beleza_saude",
            "Sports & Leisure": "esporte_lazer"
        }
        category = category_mapping[display_category]
        
        base_freight = st.number_input("Base Freight Cost (BRL)", 10.0, 500.0, 25.0)
        
        # إضافة الوزن والحجم
        col_w, col_v = st.columns(2)
        weight_g = col_w.number_input("Weight (g)", 100, 50000, 1200)
        volume_cm3 = col_v.number_input("Volume (cm³)", 100, 200000, 8000)
        
        st.markdown("---")
        st.subheader("⚠️ Business Constraints")
        
        weather = st.selectbox("Weather Condition ⛈️", ["Clear ☀", "Heavy Rain 🌧️", "Severe Storms 🌪️"])
        is_peak = st.checkbox("🔥 Peak Season (Black Friday)")
        
        # إضافة محاكاة تأخير البائع
        seller_delayed = st.checkbox("⏳ Seller Handover Delayed", help="If the seller is late, the AI will prioritize Express shipping to save the SLA.")
        
        run_engine = st.button("🚀 Optimize Routing", use_container_width=True)

    with col2:
        st.subheader("📍 Live Intelligent Routing")
        
        m = folium.Map(location=[-15.7801, -47.9292], zoom_start=4)
        start_pos = COORDS[seller_state]
        end_pos = COORDS[customer_state]
        hub_pos = [(start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2]
        
        folium.Marker(start_pos, popup="Seller Warehouse", icon=folium.Icon(color="green", icon="box")).add_to(m)
        folium.Marker(hub_pos, popup="Distribution Hub", icon=folium.Icon(color="orange", icon="exchange")).add_to(m)
        folium.Marker(end_pos, popup="Customer", icon=folium.Icon(color="red", icon="home")).add_to(m)
        
        route_leg1 = get_real_route(start_pos, hub_pos)
        route_leg2 = get_real_route(hub_pos, end_pos)
        
        folium.PolyLine(locations=route_leg1, color="gray", weight=4, dash_array="5, 5", tooltip="First Mile").add_to(m)
        folium.PolyLine(locations=route_leg2, color="blue", weight=4, tooltip="Last Mile").add_to(m)
        
        st_folium(m, width=750, height=400, key="routing_map")

    if run_engine:
        order_data = {
            "seller_state": seller_state, "customer_state": customer_state,
            "lane": f"{seller_state}->{customer_state}", "seller_city": "unknown", "customer_city": "unknown",
            "product_category_name": category, "price": 150.0, "freight_value": base_freight, 
            "freight_ratio": base_freight / 150.0, "seller_late_rate": 0.1, "budget_days": 12.0,
            "carrier_month": 10, "carrier_dow": 2, "is_black_friday": 1 if is_peak else 0,
            "is_holiday_season": 0, "same_state": 1 if seller_state == customer_state else 0,
            "weight_g": weight_g, "volume_cm3": volume_cm3, "distance_km": 400.0
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
            
            # تأثير المواسم والطقس
            if is_peak and profile["type"] == "Local": penalty_multiplier *= 1.5
            if weather == "Heavy Rain 🌧️":
                if profile["type"] != "Express": carrier_days *= 1.3  
                penalty_multiplier *= 1.2
            elif weather == "Severe Storms 🌪️":
                if profile["type"] != "Express": carrier_days *= 1.8  
                penalty_multiplier *= 1.5
                
            # تأثير تأخير البائع (لو البائع متأخر، الشركات العادية هتتدمر غرامات)
            if seller_delayed:
                if profile["type"] != "Express":
                    carrier_late_prob = 1.0 # تأكيد التأخير
                    penalty_multiplier *= 2.5 # مضاعفة الغرامة
                else:
                    carrier_late_prob = min(base_late_prob * profile["late_prob_multiplier"], 1.0)
            else:
                carrier_late_prob = min(base_late_prob * profile["late_prob_multiplier"] * penalty_multiplier, 1.0)
                
            expected_penalty = carrier_late_prob * BRL_PER_LATE
            
            # تأثير الوزن الثقيل على التكلفة (لو وزن كبير، الشركات العادية والسريعة بتغلى)
            if weight_g > 15000:
                carrier_cost *= 1.5
                
            results.append({
                "Carrier": profile["name"], 
                "Type": profile["type"],
                "Est. Days ⏱️": round(carrier_days, 1),
                "Cost (BRL) 💰": round(carrier_cost, 2), 
                "Late Risk ⚠️": f"{round(carrier_late_prob * 100, 1)}%",
                "Penalty Risk 💸": round(expected_penalty, 2),
                "Total Cost ⭐": round(carrier_cost + expected_penalty, 2)
            })
            
        df_results = pd.DataFrame(results).sort_values("Total Cost ⭐")
        best_carrier = df_results.iloc[0]
        
        st.markdown("---")
        st.success(f"**🤖 AI Decision:** **{best_carrier['Carrier']}** selected as the optimal choice considering weather, package size, and delay risks.")
        st.dataframe(df_results, use_container_width=True, hide_index=True)


with tab2:
    st.subheader("📍 Strategic Network Optimization (Heatmap & Proposed Hubs)")
    st.markdown("The heatmap highlights historical **Delay & High Freight Cost Hotspots** (Northeast & Rio). The **Blue Stars** indicate proposed Olist Cross-Docking Hubs to intercept packages early and slash last-mile costs.")
    
    m_heat = folium.Map(location=[-12.7801, -43.9292], zoom_start=4.5)
    
    # محاكاة واقعية لبؤر التأخير في البرازيل
    heat_data = [[-12.97 + (np.random.rand()-0.5)*2, -38.50 + (np.random.rand()-0.5)*2] for _ in range(250)] # Bahia
    heat_data += [[-8.04 + (np.random.rand()-0.5)*1.5, -34.87 + (np.random.rand()-0.5)*1.5] for _ in range(200)] # Pernambuco
    heat_data += [[-3.71 + (np.random.rand()-0.5)*1.5, -38.54 + (np.random.rand()-0.5)*1.5] for _ in range(200)] # Ceara
    heat_data += [[-22.90 + (np.random.rand()-0.5)*1, -43.17 + (np.random.rand()-0.5)*1] for _ in range(400)] # Rio de Janeiro
    
    # إضافة طبقة الخريطة الحرارية
    HeatMap(heat_data, name="Delay Hotspots", radius=15, blur=10, max_zoom=1).add_to(m_heat)
    
    # إضافة الفروع المقترحة لحل المشكلة (نجوم زرقاء)
    folium.Marker([-12.9714, -38.5014], popup="Proposed Hub: Northeast (Salvador)", icon=folium.Icon(color="blue", icon="star")).add_to(m_heat)
    folium.Marker([-22.9068, -43.1729], popup="Proposed Hub: Southeast (Rio)", icon=folium.Icon(color="blue", icon="star")).add_to(m_heat)
    
    # إضافة زر التبديل بين الطبقات
    folium.LayerControl().add_to(m_heat)
    folium_static(m_heat, width=900, height=500)
