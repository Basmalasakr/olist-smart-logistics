import streamlit as st
import pandas as pd
import joblib
import json
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Olist AI Logistics", page_icon="🗺️", layout="wide")

# --- إحداثيات الولايات لرسم الخريطة ---
COORDS = {
    "SP": [-23.5505, -46.6333], "RJ": [-22.9068, -43.1729],
    "MG": [-19.9167, -43.9345], "RS": [-30.0346, -51.2177],
    "PR": [-25.4284, -49.2733], "BA": [-12.9714, -38.5014],
    "CE": [-3.7172, -38.5431]
}

@st.cache_resource
def load_assets():
    models = joblib.load("models/shipping_models_advanced.joblib")
    with open("models/penalty.json", "r") as f:
        penalty = json.load(f)
    with open("models/carriers.json", "r") as f:
        carriers = json.load(f)
    return models, penalty["penalty_cost_brl"], carriers

models, BRL_PER_LATE, carriers = load_assets()

st.title("🗺️ Olist Smart Logistics Engine")
st.markdown("محرك تحسين الشحن المدعوم بالذكاء الاصطناعي لاختيار أفضل مسار وتكلفة.")

# --- واجهة الإدخال ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📦 بيانات الشحنة")
    seller_state = st.selectbox("ولاية البائع", ["SP", "RJ", "MG", "RS", "PR"])
    customer_state = st.selectbox("ولاية العميل", ["RJ", "SP", "MG", "BA", "CE"])
    category = st.selectbox("فئة المنتج", ["beleza_saude", "cama_mesa_banho", "esporte_lazer", "informatica_acessorios"])
    price = st.number_input("سعر المنتج (BRL)", 10.0, 5000.0, 150.0)
    base_freight = st.number_input("تكلفة الشحن الأساسية", 10.0, 200.0, 25.0)
    seller_late_rate = st.slider("معدل تأخير البائع التاريخي (%)", 0, 100, 10) / 100.0
    budget_days = st.slider("الأيام المسموحة للتوصيل", 3, 30, 12)

    run_engine = st.button("🚀 تشغيل المحرك الذكي", use_container_width=True)

with col2:
    st.subheader("📍 مسار الرحلة")
    # رسم الخريطة
    m = folium.Map(location=[-15.7801, -47.9292], zoom_start=4)
    start_pos = COORDS[seller_state]
    end_pos = COORDS[customer_state]
    
    folium.Marker(start_pos, popup="البائع", icon=folium.Icon(color="green", icon="box")).add_to(m)
    folium.Marker(end_pos, popup="العميل", icon=folium.Icon(color="red", icon="home")).add_to(m)
    
    # رسم خط السير
    folium.PolyLine(locations=[start_pos, end_pos], color="blue", weight=3, dash_array="5, 5").add_to(m)
    
    st_folium(m, width=700, height=350)

# --- تشغيل المحرك ---
if run_engine:
    st.markdown("---")
    st.subheader("🏆 أفضل خيارات الشحن (مقارنة ذكية)")
    
    # تجهيز الداتا للموديل الجديد
    order_data = {
        "seller_state": seller_state, "customer_state": customer_state,
        "lane": f"{seller_state}->{customer_state}", 
        "seller_city": "unknown", "customer_city": "unknown", # تبسيط للـ Demo
        "product_category_name": category, "price": price,
        "freight_value": base_freight, "freight_ratio": base_freight / price,
        "seller_late_rate": seller_late_rate, "budget_days": float(budget_days),
        "carrier_month": 10, "carrier_dow": 2, "is_black_friday": 0,
        "is_holiday_season": 0, "same_state": 1 if seller_state == customer_state else 0,
        "weight_g": 1200, "volume_cm3": 8000, "distance_km": 400.0 # قيم افتراضية
    }

    df_order = pd.DataFrame([order_data])
    for col in models["categories"]:
        if col in df_order.columns: df_order[col] = df_order[col].astype("category")
            
    base_days = models["model_p50"].predict(df_order[models["features"]])[0]
    base_late_prob = models["model_late"].predict_proba(df_order[models["features"]])[0, 1]
    
    results = []
    for carrier_id, profile in carriers.items():
        carrier_days = base_days * profile["speed_multiplier"]
        carrier_cost = base_freight * profile["cost_multiplier"]
        carrier_late_prob = min(base_late_prob * profile["late_prob_multiplier"], 1.0)
        expected_penalty = carrier_late_prob * BRL_PER_LATE
        
        results.append({
            "الشركة": profile["name"], "النوع": profile["type"],
            "الأيام المتوقعة ⏱️": round(carrier_days, 1),
            "تكلفة الشحن 💰": round(carrier_cost, 2),
            "خطر التأخير ⚠️": f"{round(carrier_late_prob * 100, 1)}%",
            "غرامة متوقعة 💸": round(expected_penalty, 2),
            "التكلفة الإجمالية ⭐": round(carrier_cost + expected_penalty, 2)
        })
        
    df_results = pd.DataFrame(results).sort_values("التكلفة الإجمالية ⭐")
    st.dataframe(df_results, use_container_width=True, hide_index=True)
