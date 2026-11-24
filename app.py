import json, urllib.request, urllib.parse
import numpy as np, pandas as pd, streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="GlobalQuake AI | Seismic & Climate Risk", page_icon="🌍", layout="centered")

# Sidebar for Custom Background Image & Social Links Setup
with st.sidebar:
    st.header("🎨 Customize Appearance")
    bg_url = st.text_input(
        "Custom Background Image URL",
        value="https://unsplash.com",
        help="Paste any direct image link (.jpg/.png) to change the website background."
    )
    st.markdown("---")
    st.header("🔗 Edit Profile Links")
    gh_user = st.text_input("GitHub Username", value="Shankar-tgs")
    tw_user = st.text_input("X (Twitter) Handle", value="7shankar18")
    li_url = st.text_input("LinkedIn Profile URL", value="https://linkedin.com")
    contact_email = st.text_input("Contact Email", value="tangudub261322ce@nitc.ac.in")

# 1. Global CSS: 3-Second Entrance Quake Animation, Custom Background, Tile Shadows & Footer
st.markdown(f"""<style>
@keyframes entranceQuake {{
    0% {{ transform: translate(0, 0) rotate(0deg); }}
    10% {{ transform: translate(-8px, -4px) rotate(-1.2deg); }}
    20% {{ transform: translate(9px, 3px) rotate(1.2deg); }}
    30% {{ transform: translate(-9px, 4px) rotate(-1deg); }}
    40% {{ transform: translate(8px, -3px) rotate(1deg); }}
    50% {{ transform: translate(-6px, 2px) rotate(-0.8deg); }}
    60% {{ transform: translate(6px, -2px) rotate(0.8deg); }}
    70% {{ transform: translate(-4px, 1px) rotate(-0.5deg); }}
    80% {{ transform: translate(4px, -1px) rotate(0.5deg); }}
    90% {{ transform: translate(-2px, 1px) rotate(-0.2deg); }}
    100% {{ transform: translate(0, 0) rotate(0deg); }}
}}
.stApp {{
    background: linear-gradient(rgba(9, 13, 22, 0.84), rgba(15, 23, 42, 0.90)), url("{bg_url}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: #f8fafc;
    font-family: 'Segoe UI', sans-serif;
}}
.main .block-container {{
    animation: entranceQuake 3s ease-out 1;
}}
.hero {{
    background: linear-gradient(90deg, rgba(30,58,138,0.92), rgba(3,105,161,0.92), rgba(13,148,136,0.92));
    padding: 26px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.65);
    border: 1px solid rgba(255,255,255,0.18);
}}
.hero h1 {{ margin: 0; font-size: 2rem; font-weight: 800; }}
.tile-card {{
    background: rgba(17, 24, 39, 0.85);
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.25);
    margin-bottom: 16px;
}}
.sec {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #38bdf8;
    margin: 8px 0 12px;
    border-bottom: 1px solid rgba(56,189,248,0.3);
    padding-bottom: 4px;
}}
label, .stMarkdown p {{ color: #e2e8f0 !important; }}
div.stButton > button {{
    width: 100%;
    background: linear-gradient(90deg, #f59e0b, #ea580c);
    color: white;
    border: none;
    padding: 13px;
    font-size: 1.1rem;
    font-weight: bold;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(234, 88, 12, 0.45);
}}
.g1 {{ background: rgba(6, 78, 59, 0.92); border-left: 7px solid #10b981; color: #ecfdf5; padding: 18px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.55); }}
.g2 {{ background: rgba(120, 53, 15, 0.92); border-left: 7px solid #f59e0b; color: #fffbeb; padding: 18px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.55); }}
.g3 {{ background: rgba(127, 29, 29, 0.92); border-left: 7px solid #ef4444; color: #fef2f2; padding: 18px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.55); }}
.footer-box {{
    background: rgba(15, 23, 42, 0.92);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    margin-top: 30px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.25);
}}
.social-links a {{
    display: inline-block;
    margin: 6px 10px;
    padding: 8px 16px;
    background: rgba(30, 41, 59, 0.95);
    color: #38bdf8 !important;
    text-decoration: none;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid rgba(56, 189, 248, 0.35);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}}
</style>""", unsafe_allow_html=True)

# 2. Train & Cache Global + Climate-Aware Random Forest Model
@st.cache_resource
def load_model():
    rng = np.random.RandomState(42)
    n = 6000
    g1, g2, g3 = rng.randint(0, 31, n), rng.randint(0, 1428, n), rng.randint(0, 12568, n)
    haz, mag, soil = rng.randint(1, 4, n), rng.uniform(4.5, 8.8, n), rng.randint(0, 3, n)
    rain_wetness = rng.uniform(0, 1, n)
    age, area, ht, flr = rng.randint(0, 120, n), rng.randint(1, 50, n), rng.randint(2, 25, n), rng.randint(1, 8, n)
    fnd, roof = rng.randint(0, 4, n), rng.randint(0, 3, n)
    risk = (
        (mag - 5.0)*0.85 + (haz - 1)*0.45 + (soil == 2)*0.70 + (soil == 1)*0.30
        + rain_wetness*0.35 + (g1 > 15)*0.30 + (age > 25)*0.55 + (age > 55)*0.55
        + (flr >= 3)*0.45 + (ht/(area+1) > 1.2)*0.40 + (fnd == 3)*1.15
        + (fnd == 2)*0.55 + (fnd == 1)*0.25 + (roof == 2)*0.75 + (roof == 1)*0.30
        + rng.normal(0, 0.35, n)
    )
    y = np.where(risk < 2.3, 1, np.where(risk < 4.2, 2, 3))
    X = pd.DataFrame({
        "g1": g1, "g2": g2, "g3": g3, "haz": haz, "mag": mag, "soil": soil,
        "wet": rain_wetness, "age": age, "area": area, "ht": ht, "flr": flr, "fnd": fnd, "roof": roof
    })
    return RandomForestClassifier(n_estimators=55, random_state=42).fit(X, y)

# 3. Live Worldwide Geocoding, Climate & Seismic APIs
@st.cache_data(ttl=1800)
def fetch_world_place_data(place_name):
    try:
        q = urllib.parse.quote(place_name)
        geo_url = f"https://open-meteo.com{q}&count=1&language=en&format=json"
        geo_res = json.loads(urllib.request.urlopen(geo_url, timeout=6).read().decode())
        results = geo_res.get("results", [])
        if not results:
            return None
        loc = results
        lat, lon = float(loc["latitude"]), float(loc["longitude"])
        full_name = f"{loc.get('name', place_name)}, {loc.get('country', '')}"

        # Live Climate Conditions (Open-Meteo Weather API)
        wx_url = f"https://open-meteo.com{lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        wx_res = json.loads(urllib.request.urlopen(wx_url, timeout=6).read().decode()).get("current", {})

        # Nearby Historical Seismic Activity (USGS 300km radius, M4.0+)
        usgs_url = f"https://usgs.gov{lat}&longitude={lon}&maxradiuskm=300&minmagnitude=4.0&limit=5"
        eq_res = json.loads(urllib.request.urlopen(usgs_url, timeout=6).read().decode()).get("features", [])
        max_eq = max([float(f["properties"]["mag"]) for f in eq_res], default=5.2)
        hazard = 3 if (len(eq_res) >= 3 or max_eq >= 6.5) else (2 if len(eq_res) >= 1 else 1)

        return {
            "name": full_name, "lat": lat, "lon": lon,
            "temp": wx_res.get("temperature_2m", 26.0),
            "humidity": wx_res.get("relative_humidity_2m", 65),
            "precip": wx_res.get("precipitation", 0.0),
            "wind": wx_res.get("wind_speed_10m", 10.0),
            "quake_count": len(eq_res), "max_eq": round(max_eq, 1), "hazard": hazard
        }
    except Exception:
        return None

@st.cache_data(ttl=1800)
def get_global_feed():
    try:
        url = "https://usgs.gov"
        data = json.loads(urllib.request.urlopen(url, timeout=8).read().decode())
        rows = []
        for f in data.get("features", [])[:40]:
            lon, lat, depth = f["geometry"]["coordinates"]
            rows.append({
                "Location": f["properties"]["place"],
                "Magnitude": round(float(f["properties"]["mag"]), 1),
                "Depth (km)": round(float(depth), 1),
                "lat": float(lat), "lon": float(lon)
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# 4. Interactive Building Animation Component (Customizes Floors, Roof, Material & Multi-Magnitude Shaking)
def render_building_simulator(floors, roof_code, fnd_code, age, grade):
    floor_blocks = "".join([
        f'<div class="floor" style="background:{"#64748b" if fnd_code==0 else ("#94a3b8" if fnd_code==1 else ("#b45309" if fnd_code==2 else "#78350f"))};">'
        f'<div class="win"></div><div class="win"></div><div class="win"></div>'
        f'{"<div class=crack></div>" if (age > 35 or grade >= 2) else ""}</div>'
        for _ in range(min(floors, 8))
    ])
    roof_html = (
        '<div class="roof-flat"></div>' if roof_code == 0 else
        ('<div class="roof-metal"></div>' if roof_code == 1 else '<div class="roof-pitched"></div>')
    )
    html_sim = f"""
    <div style="background:#0f172a; padding:18px; border-radius:14px; border:1px solid #334155; color:white; font-family:sans-serif; text-align:center;">
      <div style="font-weight:700; color:#38bdf8; margin-bottom:10px;">🏗️ Live Structural Simulator ({floors} Floors | Grade {grade})</div>
      <div style="margin-bottom:14px;">
        <button onclick="setMag('calm')" style="padding:6px 12px; margin:3px; border:none; border-radius:6px; background:#334155; color:white; cursor:pointer;">Normal State</button>
        <button onclick="setMag('m5')" style="padding:6px 12px; margin:3px; border:none; border-radius:6px; background:#059669; color:white; cursor:pointer;">Mw 5.0 (Moderate)</button>
