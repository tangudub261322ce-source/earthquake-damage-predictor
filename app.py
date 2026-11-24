import base64, json, urllib.request, urllib.parse
import numpy as np, pandas as pd, streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="GlobalQuake AI | Seismic & Climate Risk", page_icon="🌍", layout="centered")

# Sidebar: Upload Custom Background & Edit Follow Us Links
with st.sidebar:
    st.header("🎨 Customize Website")
    uploaded_bg = st.file_uploader("Upload Custom Background Image", type=["png", "jpg", "jpeg", "webp"])
    default_bg = "https://unsplash.com"
    if uploaded_bg is not None:
        b64 = base64.b64encode(uploaded_bg.read()).decode()
        bg_source = f"data:{uploaded_bg.type};base64,{b64}"
    else:
        bg_source = default_bg

    st.markdown("---")
    st.header("🔗 Follow Us & Contact Links")
    gh_url = st.text_input("GitHub URL", value="https://github.com")
    tw_url = st.text_input("Twitter / X URL", value="https://twitter.com")
    li_url = st.text_input("LinkedIn URL", value="https://linkedin.com")
    email_addr = st.text_input("Contact Email", value="tangudub261322ce@nitc.ac.in")

# 1. CSS: 3-Second Entry Earthquake Animation, Custom Background & Tile Shadows
css = (
    "<style>"
    "@keyframes entryQuake {"
    " 0%, 100% { transform: translate(0, 0) rotate(0deg); }"
    " 10% { transform: translate(-9px, -4px) rotate(-1.3deg); }"
    " 20% { transform: translate(9px, 4px) rotate(1.3deg); }"
    " 30% { transform: translate(-8px, 3px) rotate(-1deg); }"
    " 40% { transform: translate(8px, -3px) rotate(1deg); }"
    " 50% { transform: translate(-6px, 2px) rotate(-0.8deg); }"
    " 60% { transform: translate(6px, -2px) rotate(0.8deg); }"
    " 70% { transform: translate(-4px, 1px) rotate(-0.4deg); }"
    " 80% { transform: translate(4px, -1px) rotate(0.4deg); }"
    " 90% { transform: translate(-2px, 0px) rotate(-0.2deg); }"
    "}"
    f'.stApp {{ background: linear-gradient(rgba(9,13,22,0.82), rgba(15,23,42,0.90)), url("{bg_source}"); background-size: cover; background-attachment: fixed; background-position: center; color: #f8fafc; font-family: "Segoe UI", sans-serif; }}'
    '[data-testid="stMainBlockContainer"] { animation: entryQuake 3s ease-out 1; }'
    ".hero { background: linear-gradient(90deg, rgba(30,58,138,0.92), rgba(3,105,161,0.92), rgba(13,148,136,0.92)); padding: 24px; border-radius: 16px; text-align: center; color: white; margin-bottom: 18px; box-shadow: 0 14px 32px rgba(0,0,0,0.7); border: 1px solid rgba(255,255,255,0.18); }"
    ".hero h1 { margin: 0; font-size: 1.95rem; font-weight: 800; }"
    ".tile { background: rgba(17,24,39,0.88); padding: 18px; border-radius: 14px; box-shadow: 0 12px 28px rgba(0,0,0,0.65); border: 1px solid rgba(56,189,248,0.28); margin-bottom: 16px; }"
    ".sec { font-size: 1.05rem; font-weight: 700; color: #38bdf8; margin: 6px 0 12px; border-bottom: 1px solid rgba(56,189,248,0.3); padding-bottom: 4px; }"
    "label, .stMarkdown p { color: #e2e8f0 !important; }"
    "div.stButton > button { width: 100%; background: linear-gradient(90deg, #f59e0b, #ea580c); color: white; border: none; padding: 13px; font-size: 1.08rem; font-weight: bold; border-radius: 11px; box-shadow: 0 8px 20px rgba(234,88,12,0.5); }"
    ".g1 { background: rgba(6,78,59,0.94); border-left: 7px solid #10b981; color: #ecfdf5; padding: 18px; border-radius: 12px; box-shadow: 0 12px 28px rgba(0,0,0,0.6); }"
    ".g2 { background: rgba(120,53,15,0.94); border-left: 7px solid #f59e0b; color: #fffbeb; padding: 18px; border-radius: 12px; box-shadow: 0 12px 28px rgba(0,0,0,0.6); }"
    ".g3 { background: rgba(127,29,29,0.94); border-left: 7px solid #ef4444; color: #fef2f2; padding: 18px; border-radius: 12px; box-shadow: 0 12px 28px rgba(0,0,0,0.6); }"
    ".footer { background: rgba(15,23,42,0.94); padding: 20px; border-radius: 16px; text-align: center; margin-top: 26px; box-shadow: 0 14px 32px rgba(0,0,0,0.7); border: 1px solid rgba(148,163,184,0.25); }"
    ".slink a { display: inline-block; margin: 5px 8px; padding: 8px 14px; background: rgba(30,41,59,0.95); color: #38bdf8 !important; text-decoration: none; border-radius: 8px; font-weight: 600; border: 1px solid rgba(56,189,248,0.35); box-shadow: 0 4px 12px rgba(0,0,0,0.45); }"
    "</style>"
)
st.markdown(css, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    rng = np.random.RandomState(42)
    n = 5500
    g1, g2, g3 = rng.randint(0, 31, n), rng.randint(0, 1428, n), rng.randint(0, 12568, n)
    haz, mag, soil = rng.randint(1, 4, n), rng.uniform(4.5, 8.8, n), rng.randint(0, 3, n)
    wet = rng.uniform(0, 1, n)
    age, area, ht, flr = rng.randint(0, 120, n), rng.randint(1, 50, n), rng.randint(2, 25, n), rng.randint(1, 8, n)
    fnd, roof = rng.randint(0, 4, n), rng.randint(0, 3, n)
    risk = (
        (mag - 5.0)*0.85 + (haz - 1)*0.45 + (soil == 2)*0.70 + (soil == 1)*0.30
        + wet*0.35 + (g1 > 15)*0.30 + (age > 25)*0.55 + (age > 55)*0.55
        + (flr >= 3)*0.45 + (ht/(area+1) > 1.2)*0.40 + (fnd == 3)*1.15
        + (fnd == 2)*0.55 + (fnd == 1)*0.25 + (roof == 2)*0.75 + (roof == 1)*0.30
        + rng.normal(0, 0.35, n)
    )
    y = np.where(risk < 2.3, 1, np.where(risk < 4.2, 2, 3))
    X = pd.DataFrame({"g1": g1, "g2": g2, "g3": g3, "haz": haz, "mag": mag, "soil": soil, "wet": wet, "age": age, "area": area, "ht": ht, "flr": flr, "fnd": fnd, "roof": roof})
    return RandomForestClassifier(n_estimators=55, random_state=42).fit(X, y)

@st.cache_data(ttl=1800)
def fetch_world_place(place_name):
    try:
        q = urllib.parse.quote(place_name)
        geo_url = f"https://open-meteo.com{q}&count=1&language=en&format=json"
        geo_res = json.loads(urllib.request.urlopen(geo_url, timeout=6).read().decode())
        loc = next(iter(geo_res.get("results", [])), None)
        if not loc:
            return None
        lat, lon = float(loc["latitude"]), float(loc["longitude"])
        full_name = f"{loc.get('name', place_name)}, {loc.get('country', '')}"
        wx_url = f"https://open-meteo.com{lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        wx = json.loads(urllib.request.urlopen(wx_url, timeout=6).read().decode()).get("current", {})
        usgs_url = f"https://usgs.gov{lat}&longitude={lon}&maxradiuskm=300&minmagnitude=4.0&limit=5"
        eqs = json.loads(urllib.request.urlopen(usgs_url, timeout=6).read().decode()).get("features", [])
        max_eq = max([float(f["properties"]["mag"]) for f in eqs], default=5.2)
        haz = 3 if (len(eqs) >= 3 or max_eq >= 6.5) else (2 if len(eqs) >= 1 else 1)
        return {
            "name": full_name, "lat": lat, "lon": lon,
            "temp": wx.get("temperature_2m", 26.0), "humidity": wx.get("relative_humidity_2m", 65),
            "precip": wx.get("precipitation", 0.0), "wind": wx.get("wind_speed_10m", 10.0),
            "quakes": len(eqs), "max_eq": round(max_eq, 1), "haz": haz
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
            rows.append({"Location": f["properties"]["place"], "Magnitude": round(float(f["properties"]["mag"]), 1), "Depth (km)": round(float(depth), 1), "lat": float(lat), "lon": float(lon)})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def get_building_html(floors, roof_code, fnd_code, age, grade, default_mode="m7", compact=False):
    wall_col = "#64748b" if fnd_code == 0 else ("#94a3b8" if fnd_code == 1 else ("#b45309" if fnd_code == 2 else "#78350f"))
    crack_html = '<div class="crack"></div>' if (age > 35 or grade >= 2) else ""
    floor_divs = "".join([f'<div class="flr" style="background:{wall_col};"><div class="win"></div><div class="win"></div><div class="win"></div>{crack_html}</div>' for _ in range(min(floors, 8))])
    roof_div = '<div class="r-flat"></div>' if roof_code == 0 else ('<div class="r-metal"></div>' if roof_code == 1 else '<div class="r-pitch"></div>')
    controls = "" if compact else (
        '<div style="margin-bottom:10px;">'
        '<button onclick="setMag(\'calm\')" style="padding:5px 10px;margin:2px;border:none;border-radius:6px;background:#334155;color:white;cursor:pointer;">Normal</button>'
        '<button onclick="setMag(\'m5\')" style="padding:5px 10px;margin:2px;border:none;border-radius:6px;background:#059669;color:white;cursor:pointer;">Mw 5.0</button>'
        '<button onclick="setMag(\'m7\')" style="padding:5px 10px;margin:2px;border:none;border-radius:6px;background:#d97706;color:white;cursor:pointer;">Mw 7.0</button>'
        '<button onclick="setMag(\'m8\')" style="padding:5px 10px;margin:2px;border:none;border-radius:6px;background:#dc2626;color:white;cursor:pointer;">Mw 8.5</button>'
        '</div>'
    )
    anim_init = "shake5 0.35s infinite" if default_mode == "m5" else ("shake8 0.14s infinite" if default_mode == "m8" else "shake7 0.22s infinite")
    box_h = 185 if compact else 225
    return (
        f'<div style="background:#0f172a;padding:14px;border-radius:14px;border:1px solid #334155;color:white;font-family:sans-serif;text-align:center;box-shadow:0 10px 25px rgba(0,0,0,0.6);">'
        f'<div style="font-weight:700;color:#38bdf8;margin-bottom:8px;">🏗️ Custom Building Seismic Animation ({floors} Floors | Grade {grade})</div>'
        f'{controls}'
    )
