import numpy as np, pandas as pd, streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Use layout="wide" so our CSS @media queries control exact XS/SM/MD vs LG/XL widths
st.set_page_config(page_title="GlobalQuake AI", page_icon="🌍", layout="wide")

# =========================================================================
# 1. TWO-MODE RESPONSIVE CSS (XS/SM/MD <= 991px  |  LG/XL >= 992px)
# =========================================================================
st.markdown("""<style>
.stApp {
    background: linear-gradient(rgba(9, 13, 22, 0.82), rgba(15, 23, 42, 0.90)),
                url("https://unsplash.com");
    background-size: cover;
    background-position: center;
    color: #f8fafc;
    font-family: 'Segoe UI', sans-serif;
}
.hero {
    background: linear-gradient(90deg, #1e3a8a, #0369a1, #0d9488);
    text-align: center;
    color: white;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.6);
}
.hero h1 { margin: 0; font-weight: 800; line-height: 1.2; }
.hero p { margin: 8px 0 0; opacity: 0.92; }
.sec {
    font-weight: 700;
    color: #38bdf8;
    margin: 10px 0;
    border-bottom: 1px solid rgba(56,189,248,0.3);
    padding-bottom: 4px;
}
label, .stMarkdown p { color: #e2e8f0 !important; }
[data-testid="stMetric"] {
    background-color: rgba(30, 41, 59, 0.88);
    border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.3);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.45);
}
button[data-baseweb="tab"] { font-weight: 600; color: #94a3b8; }
button[data-baseweb="tab"][aria-selected="true"] { color: #38bdf8 !important; }
div.stButton > button {
    width: 100%;
    min-height: 48px;
    background: linear-gradient(90deg, #f59e0b, #ea580c);
    color: white;
    border: none;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(234, 88, 12, 0.45);
}
.card { border-radius: 12px; margin: 14px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.55); }
.g1 { background: #064e3b; border-left: 6px solid #10b981; color: #ecfdf5; }
.g2 { background: #78350f; border-left: 6px solid #f59e0b; color: #fffbeb; }
.g3 { background: #7f1d1d; border-left: 6px solid #ef4444; color: #fef2f2; }
.footer-card {
    background: rgba(30, 41, 59, 0.92);
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    margin-top: 24px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}

/* =========================================================================
   TYPE 1: EXTRA-SMALL, SMALL & MEDIUM SCREENS (<= 991px) — STACKED MOBILE/TABLET
   ========================================================================= */
@media only screen and (max-width: 991px) {
    .stApp { background-attachment: scroll; }
    .block-container {
        max-width: 100% !important;
        padding: 1.1rem 0.85rem 1.5rem 0.85rem !important;
    }
    .hero { padding: 18px 14px; border-radius: 12px; margin-bottom: 14px; }
    .hero h1 { font-size: 1.4rem !important; }
    .hero p { font-size: 0.88rem !important; }
    .sec { font-size: 0.98rem !important; }
    [data-baseweb="tab-list"] { overflow-x: auto; white-space: nowrap; gap: 4px; }
    button[data-baseweb="tab"] { font-size: 0.88rem !important; padding: 8px 10px !important; }
    /* Force single-column stacking on XS, SM, and MD */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.6rem !important;
    }
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    [data-testid="stMetric"] { padding: 10px 14px !important; }
    .card { padding: 14px; }
    div.stButton > button { font-size: 1rem !important; padding: 11px !important; }
}

/* =========================================================================
   TYPE 2: LARGE & EXTRA-LARGE SCREENS (>= 992px) — WIDESCREEN DESKTOP GRID
   ========================================================================= */
@media only screen and (min-width: 992px) {
    .stApp { background-attachment: fixed; }
    .block-container {
        max-width: 1180px !important;
        margin: 0 auto !important;
        padding: 2rem 2.5rem 2.5rem 2.5rem !important;
    }
    .hero { padding: 30px 24px; border-radius: 18px; margin-bottom: 22px; }
    .hero h1 { font-size: 2.25rem !important; }
    .hero p { font-size: 1.08rem !important; }
    .sec { font-size: 1.12rem !important; }
    button[data-baseweb="tab"] { font-size: 1.05rem !important; padding: 10px 18px !important; }
    /* Side-by-side multi-column layout on LG and XL */
    [data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        gap: 1.5rem !important;
    }
    [data-testid="stMetric"] { padding: 16px 18px !important; }
    .card { padding: 20px 24px; }
    div.stButton > button { font-size: 1.15rem !important; padding: 14px !important; }
}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_calibrated_model():
    rng = np.random.RandomState(42)
    n = 8000
    haz = rng.randint(1, 4, n)
    mag = rng.uniform(4.5, 9.0, n)
    soil = rng.randint(0, 3, n)
    age = rng.randint(0, 150, n)
    area = rng.randint(1, 100, n)
    ht = rng.randint(2, 32, n)
    flr = rng.randint(1, 10, n)
    fnd = rng.randint(0, 4, n)
    roof = rng.randint(0, 3, n)

    risk = (
        (mag - 4.5) * 0.70 +
        (haz - 1) * 0.35 +
        soil * 0.45 +
        (age / 50.0) * 0.65 +
        (flr / 4.0) * 0.45 +
        (ht / (area + 5.0)) * 0.35 +
        fnd * 0.65 +
        roof * 0.45 +
        rng.normal(0, 0.25, n)
    )
    y = np.where(risk < 3.2, 1, np.where(risk < 5.3, 2, 3))

    X = pd.DataFrame({
        "haz": haz, "mag": mag, "soil": soil, "age": age,
        "area": area, "ht": ht, "flr": flr, "fnd": fnd, "roof": roof
    })
    clf = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42)
    clf.fit(X, y)
    return clf

@st.cache_data(ttl=600)
def fetch_global_quakes():
    try:
        df = pd.read_csv("https://usgs.gov")
        clean = df[["place", "mag", "depth", "latitude", "longitude"]].head(40).copy()
        clean.columns = ["Location", "Magnitude", "Depth (km)", "lat", "lon"]
        return clean.dropna()
    except Exception:
        return pd.DataFrame([
            {"Location": "Kathmandu Valley, Nepal", "Magnitude": 7.8, "Depth (km)": 8.2, "lat": 27.71, "lon": 85.32},
            {"Location": "Sendai Coast, Honshu, Japan", "Magnitude": 7.4, "Depth (km)": 35.0, "lat": 38.27, "lon": 141.02},
            {"Location": "Kahramanmaras, Turkey", "Magnitude": 7.5, "Depth (km)": 10.0, "lat": 37.58, "lon": 37.20},
            {"Location": "Valparaiso Offshore, Chile", "Magnitude": 6.9, "Depth (km)": 28.4, "lat": -33.04, "lon": -71.62},
            {"Location": "Hualien County, Taiwan", "Magnitude": 7.2, "Depth (km)": 15.5, "lat": 23.99, "lon": 121.60},
            {"Location": "Sumatra Fault Zone, Indonesia", "Magnitude": 6.7, "Depth (km)": 22.1, "lat": -0.95, "lon": 100.35},
            {"Location": "San Andreas Corridor, California, USA", "Magnitude": 6.4, "Depth (km)": 11.8, "lat": 35.77, "lon": -120.40},
            {"Location": "Central Apennines, Italy", "Magnitude": 6.2, "Depth (km)": 9.5, "lat": 42.79, "lon": 13.10}
        ])

model = load_calibrated_model()
st.markdown('<div class="hero"><h1>🌍 GlobalQuake AI: Damage Predictor</h1><p>Worldwide Seismic Vulnerability & Building Risk Assessment</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Damage Predictor", "🌍 Global Seismic Feed", "📊 Insights & Info"])

with tab1:
    zones = {
        "Himalayan Belt (India / Nepal) — Extreme Hazard": (3, 17, 450, 3000),
        "Pacific Ring of Fire (Japan / Indonesia) — Extreme Hazard": (3, 24, 890, 7400),
        "Andean Belt (Chile / Peru) — Extreme Hazard": (3, 22, 760, 6100),
        "Anatolian Fault (Turkey / Iran) — High Hazard": (2, 19, 610, 4800),
        "San Andreas Fault (USA / Mexico) — High Hazard": (2, 14, 380, 2900),
        "Mediterranean Belt (Italy / Greece) — Moderate Hazard": (2, 11, 290, 2100),
        "Custom Location (Manual Region/District IDs)": (2, 17, 450, 3000)
    }
    haz, dg1, dg2, dg3 = zones[st.selectbox("Select Global Seismic Region", list(zones.keys()))]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec">📍 Location & Seismic Load</div>', unsafe_allow_html=True)
        g1 = st.number_input("Region ID (Geo 1: 0–30)", 0, 30, dg1)
        g2 = st.number_input("District ID (Geo 2: 0–1427)", 0, 1427, dg2)
        g3 = st.number_input("Municipality ID (Geo 3: 0–12567)", 0, 12567, dg3)
        mag = st.slider("Earthquake Magnitude (Mw)", 4.5, 9.0, 6.2, 0.1)
        soils = {"Soft Alluvial / Liquefaction Zone": 2, "Stiff Sedimentary Soil": 1, "Hard Bedrock / Basalt": 0}
        soil = soils[st.selectbox("Ground / Soil Type", list(soils.keys()))]
    with c2:
        st.markdown('<div class="sec">🧱 Materials & Dimensions</div>', unsafe_allow_html=True)
        fnds = {"Mud Mortar - Stone / Unreinforced Brick": 3, "Bamboo / Timber Frame": 2, "Cement Mortar - Brick": 1, "Reinforced Concrete (RCC)": 0}
        fnd = fnds[st.selectbox("Foundation & Structure", list(fnds.keys()))]
        roofs = {"Heavy Timber / Mud / Tile": 2, "Light Metal Sheet / Bamboo": 1, "Reinforced Concrete (RCC) Slab": 0}
        roof = roofs[st.selectbox("Roof Type", list(roofs.keys()))]
        age = st.slider("Building Age (Years)", 0, 200, 20)
        flr = st.slider("Number of Floors", 1, 9, 2)
        area = st.slider("Footprint Area %", 1, 100, 12)
        ht = st.slider("Height %", 2, 32, 6)

    if st.button("Analyze Global Structural Risk"):
        inp = pd.DataFrame([{
            "haz": haz, "mag": mag, "soil": soil, "age": age,
            "area": area, "ht": ht, "flr": flr, "fnd": fnd, "roof": roof
        }])
