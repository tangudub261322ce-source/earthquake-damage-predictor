import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="GlobalQuake AI | Worldwide Building Damage Predictor", page_icon="🌍", layout="centered")

# 2. Custom CSS Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .hero-banner {
        background: linear-gradient(90deg, #1e3a8a 0%, #0369a1 50%, #0d9488 100%);
        padding: 28px 22px;
        border-radius: 18px;
        text-align: center;
        color: white;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 22px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        margin: 8px 0 0;
        font-size: 1.02rem;
        opacity: 0.92;
    }
    .section-header {
        font-size: 1.08rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 10px;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.3);
        padding-bottom: 4px;
    }
    label, .stMarkdown p {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #f59e0b 0%, #ea580c 100%);
        color: white;
        border: none;
        padding: 14px 24px;
        font-size: 1.15rem;
        font-weight: bold;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(234, 88, 12, 0.45);
        transition: transform 0.2s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.015);
        color: white;
    }
    .result-card {
        padding: 22px;
        border-radius: 14px;
        margin-top: 18px;
        margin-bottom: 18px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
    }
    .grade-1 { background-color: #064e3b; border-left: 7px solid #10b981; color: #ecfdf5; }
    .grade-2 { background-color: #78350f; border-left: 7px solid #f59e0b; color: #fffbeb; }
    .grade-3 { background-color: #7f1d1d; border-left: 7px solid #ef4444; color: #fef2f2; }
    .info-box {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Train & Cache Global + Structural Random Forest Model
@st.cache_resource
def load_global_model():
    rng = np.random.RandomState(42)
    n_samples = 6500

    geo1 = rng.randint(0, 31, n_samples)
    geo2 = rng.randint(0, 1428, n_samples)
    geo3 = rng.randint(0, 12568, n_samples)
    # Seismic hazard zone intensity (1 = Low/Moderate, 2 = High, 3 = Extreme Ring of Fire / Himalayan)
    zone_hazard = rng.choice([1, 2, 3], size=n_samples, p=[0.25, 0.40, 0.35])
    magnitude = rng.uniform(4.5, 8.8, n_samples)
    # Soil type: 0 = Hard Bedrock, 1 = Stiff Soil, 2 = Soft Clay / Liquefaction Zone
    soil = rng.randint(0, 3, n_samples)

    age = rng.randint(0, 120, n_samples)
    area = rng.randint(1, 50, n_samples)
    height = rng.randint(2, 25, n_samples)
    floors = rng.randint(1, 8, n_samples)
    # Foundation: 0 = Reinforced Concrete, 1 = Cement/Brick, 2 = Bamboo/Timber, 3 = Mud Mortar-Stone
    foundation = rng.randint(0, 4, n_samples)
    # Roof: 0 = RCC Slab, 1 = Light Metal Sheet, 2 = Heavy Timber/Mud
    roof = rng.randint(0, 3, n_samples)

    risk_score = (
        (magnitude - 5.0) * 0.85 +
        (zone_hazard - 1) * 0.45 +
        (soil == 2).astype(int) * 0.75 +
        (soil == 1).astype(int) * 0.30 +
        (geo1 > 15).astype(int) * 0.35 +
        (age > 25).astype(int) * 0.55 +
        (age > 55).astype(int) * 0.55 +
        (floors >= 3).astype(int) * 0.45 +
        (height / (area + 1) > 1.2).astype(int) * 0.40 +
        (foundation == 3).astype(int) * 1.15 +
        (foundation == 2).astype(int) * 0.55 +
        (foundation == 1).astype(int) * 0.25 +
        (roof == 2).astype(int) * 0.75 +
        (roof == 1).astype(int) * 0.30 +
        rng.normal(0, 0.35, n_samples)
    )

    damage_grade = np.where(risk_score < 2.2, 1, np.where(risk_score < 4.1, 2, 3))

    feature_names = [
        'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
        'zone_hazard', 'magnitude', 'soil_code',
        'age', 'area_percentage', 'height_percentage',
        'count_floors_pre_eq', 'foundation_code', 'roof_code'
    ]

    X = pd.DataFrame({
        'geo_level_1_id': geo1, 'geo_level_2_id': geo2, 'geo_level_3_id': geo3,
        'zone_hazard': zone_hazard, 'magnitude': magnitude, 'soil_code': soil,
        'age': age, 'area_percentage': area, 'height_percentage': height,
        'count_floors_pre_eq': floors, 'foundation_code': foundation, 'roof_code': roof
    })

    clf = RandomForestClassifier(n_estimators=70, random_state=42)
    clf.fit(X, damage_grade)
    return clf, feature_names

# Fetch live global earthquakes from USGS (past 7 days, M4.5+)
@st.cache_data(ttl=1800)
def fetch_live_earthquakes():
    try:
        url = "https://usgs.gov"
        res = requests.get(url, timeout=6).json()
        rows = []
        for feat in res.get("features", [])[:50]:
            props = feat["properties"]
            coords = feat["geometry"]["coordinates"]
            rows.append({
                "Location": props.get("place", "Unknown"),
                "Magnitude (Mw)": round(props.get("mag", 0), 1),
                "Depth (km)": round(coords[2], 1),
                "lat": coords[1],
                "lon": coords[0]
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

model, feature_names = load_global_model()

# 4. Header Banner
st.markdown("""
    <div class="hero-banner">
        <h1>🌍 GlobalQuake AI: Worldwide Damage Predictor</h1>
        <p>Multi-Region Seismic Vulnerability & Building Risk Assessment System</p>
    </div>
""", unsafe_allow_html=True)

# 5. Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Damage Predictor", "🌍 Global Seismic Feed", "📊 Model Insights & Info"])

with tab1:
    # Global Region Selector + Existing Geo IDs
    st.markdown('<div class="section-header">🌐 Global Seismic Zone & Location IDs</div>', unsafe_allow_html=True)

    global_zones = {
        "Himalayan Belt (India / Nepal / Bhutan) — Extreme Hazard": (3, 17, 450, 3000),
        "Pacific Ring of Fire (Japan / Indonesia / Philippines) — Extreme Hazard": (3, 24, 890, 7400),
        "Andean Belt (Chile / Peru / Ecuador) — Extreme Hazard": (3, 22, 760, 6100),
        "Anatolian & Zagros Faults (Turkey / Iran) — High Hazard": (2, 19, 610, 4800),
        "San Andreas & Cascadia (USA / Canada / Mexico) — High Hazard": (2, 14, 380, 2900),
        "Mediterranean Belt (Italy / Greece) — Moderate-High Hazard": (2, 11, 290, 2100),
        "Custom Location (Use Manual Region/District IDs Below)": (2, 17, 450, 3000)
    }

    selected_zone = st.selectbox("Select Global Seismic Region", list(global_zones.keys()))
    hazard_level, def_g1, def_g2, def_g3 = global_zones[selected_zone]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📍 Regional IDs & Seismic Load</div>', unsafe_allow_html=True)
        geo1 = st.number_input("Region ID (Geo Level 1: 0–30)", min_value=0, max_value=30, value=def_g1)
        geo2 = st.number_input("District ID (Geo Level 2: 0–1427)", min_value=0, max_value=1427, value=def_g2)
        geo3 = st.number_input("Municipality ID (Geo Level 3: 0–12567)", min_value=0, max_value=12567, value=def_g3)
        magnitude = st.slider("Expected Earthquake Magnitude (Mw)", min_value=4.5, max_value=9.0, value=6.8, step=0.1)

        soil_options = {
            "Soft Alluvial Soil / Liquefaction Zone (High Amplification)": 2,
            "Stiff Sedimentary Soil (Moderate Amplification)": 1,
            "Hard Bedrock / Basalt (Low Amplification)": 0
        }
        selected_soil = st.selectbox("Local Ground / Soil Type", list(soil_options.keys()))

    with col2:
        st.markdown('<div class="section-header">🧱 Structural Materials & Dimensions</div>', unsafe_allow_html=True)
        foundation_options = {
            "Mud Mortar - Stone / Unreinforced Brick": 3,
            "Bamboo / Timber Frame": 2,
            "Cement Mortar - Brick / Masonry": 1,
            "Reinforced Concrete (RCC / Seismic Design)": 0
        }
        selected_foundation = st.selectbox("Foundation & Superstructure Type", list(foundation_options.keys()))

        roof_options = {
            "Heavy Timber / Mud / Tile Roof": 2,
            "Light Metal Sheet / Bamboo Roof": 1,
            "Reinforced Concrete (RCC) Slab": 0
        }
        selected_roof = st.selectbox("Roof Construction Type", list(roof_options.keys()))

        age = st.slider("Building Age (Years)", min_value=0, max_value=200, value=25)
        floors = st.slider("Number of Floors", min_value=1, max_value=9, value=2)
        area = st.slider("Footprint Area Percentage", min_value=1, max_value=100, value=8)
        height = st.slider("Height Percentage", min_value=2, max_value=32, value=6)

    st.write("")

    if st.button("Analyze Global Structural Risk"):
        s_code = soil_options[selected_soil]
        f_code = foundation_options[selected_foundation]
        r_code = roof_options[selected_roof]

        input_data = pd.DataFrame(
            [[geo1, geo2, geo3, hazard_level, magnitude, s_code, age, area, height, floors, f_code, r_code]],
            columns=feature_names
        )

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

