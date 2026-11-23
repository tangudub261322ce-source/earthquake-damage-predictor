import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Earthquake Damage Predictor", page_icon="🏢", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .hero-banner {
        background: linear-gradient(90deg, #2563eb 0%, #06b6d4 100%);
        padding: 25px 20px;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
        margin-bottom: 25px;
    }
    .hero-banner h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .hero-banner p { margin: 8px 0 0; font-size: 1rem; opacity: 0.92; }
    label, .stMarkdown p { color: #e2e8f0 !important; font-weight: 500; }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #f97316 0%, #ea580c 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.4);
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }
    .grade-1 { background-color: #064e3b; border-left: 6px solid #10b981; color: #ecfdf5; }
    .grade-2 { background-color: #78350f; border-left: 6px solid #f59e0b; color: #fffbeb; }
    .grade-3 { background-color: #7f1d1d; border-left: 6px solid #ef4444; color: #fef2f2; }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('earthquake_web_model.pkl')

st.markdown("""
    <div class="hero-banner">
        <h1>🏢 Earthquake Building Damage Predictor</h1>
        <p>AI-Powered Structural Risk Assessment Using 2015 Nepal Earthquake Data</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    geo1 = st.number_input("Region ID (Geo Level 1: 0–30)", min_value=0, max_value=30, value=17)
    geo2 = st.number_input("District ID (Geo Level 2: 0–1427)", min_value=0, max_value=1427, value=450)
    geo3 = st.number_input("Municipality ID (Geo Level 3: 0–12567)", min_value=0, max_value=12567, value=3000)
    floors = st.slider("Number of Floors (Before Earthquake)", min_value=1, max_value=9, value=2)

with col2:
    age = st.slider("Building Age (Years)", min_value=0, max_value=200, value=20)
    area = st.slider("Footprint Area Percentage", min_value=1, max_value=100, value=8)
    height = st.slider("Height Percentage", min_value=2, max_value=32, value=5)

st.write("")
if st.button("Analyze Structural Risk"):
    input_data = pd.DataFrame([[geo1, geo2, geo3, age, area, height, floors]],
                              columns=['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
                                       'age', 'area_percentage', 'height_percentage', 'count_floors_pre_eq'])
    prediction = model.predict(input_data)
    if prediction == 1:
        st.markdown('<div class="result-card grade-1"><h3 style="margin:0;">🟢 Grade 1: Low Damage</h3><p style="margin:8px 0 0;">Minor hairline cracks expected. Building remains structurally safe to repair.</p></div>', unsafe_allow_html=True)
    elif prediction == 2:
        st.markdown('<div class="result-card grade-2"><h3 style="margin:0;">🟡 Grade 2: Medium Damage</h3><p style="margin:8px 0 0;">Moderate wall and structural damage. Requires engineering inspection before occupancy.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card grade-3"><h3 style="margin:0;">🔴 Grade 3: Severe Damage</h3><p style="margin:8px 0 0;">High risk of partial or total structural collapse. Immediate evacuation required.</p></div>', unsafe_allow_html=True)
