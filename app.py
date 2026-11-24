import base64, json, urllib.request, urllib.parse
import numpy as np, pandas as pd, streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="GlobalQuake AI", page_icon="🌍", layout="centered")

with st.sidebar:
    st.header("🎨 Customize & Socials")
    up_bg = st.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])
    def_bg = "https://unsplash.com"
    bg_src = f"data:{up_bg.type};base64,{base64.b64encode(up_bg.read()).decode()}" if up_bg else def_bg
    gh_url = st.text_input("GitHub URL", "https://github.com")
    tw_url = st.text_input("Twitter / X URL", "https://twitter.com")
    li_url = st.text_input("LinkedIn URL", "https://linkedin.com")
    email_addr = st.text_input("Contact Email", "tangudub261322ce@nitc.ac.in")

st.markdown(f"""<style>
@keyframes entryQuake {{ 0%,100%{{transform:translate(0,0) rotate(0deg)}} 15%{{transform:translate(-9px,-4px) rotate(-1.2deg)}} 30%{{transform:translate(9px,4px) rotate(1.2deg)}} 45%{{transform:translate(-7px,3px) rotate(-0.9deg)}} 60%{{transform:translate(7px,-3px) rotate(0.9deg)}} 75%{{transform:translate(-4px,1px) rotate(-0.4deg)}} 90%{{transform:translate(4px,-1px) rotate(0.4deg)}} }}
.stApp {{ background: linear-gradient(rgba(9,13,22,0.78), rgba(15,23,42,0.88)), url("{bg_src}"); background-size: cover; background-attachment: fixed; color: #f8fafc; }}
.block-container {{ animation: entryQuake 3s ease-out 1; }}
.hero {{ background: linear-gradient(90deg, #1e3a8a, #0369a1, #0d9488); padding: 22px; border-radius: 16px; text-align: center; color: white; margin-bottom: 18px; box-shadow: 0 14px 32px rgba(0,0,0,0.8); }}
.tile {{ background: rgba(17,24,39,0.88); padding: 16px; border-radius: 14px; box-shadow: 0 12px 28px rgba(0,0,0,0.75); border: 1px solid rgba(56,189,248,0.3); margin-bottom: 14px; }}
div.stButton>button {{ width: 100%; background: linear-gradient(90deg, #f59e0b, #ea580c); color: white; border: none; padding: 12px; font-weight: bold; border-radius: 10px; box-shadow: 0 8px 20px rgba(234,88,12,0.5); }}
.g1 {{ background: #064e3b; border-left: 6px solid #10b981; padding: 16px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.7); }}
.g2 {{ background: #78350f; border-left: 6px solid #f59e0b; padding: 16px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.7); }}
.g3 {{ background: #7f1d1d; border-left: 6px solid #ef4444; padding: 16px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.7); }}
.footer {{ background: rgba(15,23,42,0.94); padding: 18px; border-radius: 14px; text-align: center; margin-top: 24px; box-shadow: 0 12px 28px rgba(0,0,0,0.8); border: 1px solid rgba(148,163,184,0.25); }}
.footer a {{ display: inline-block; margin: 4px 8px; padding: 6px 14px; background: #1e293b; color: #38bdf8 !important; text-decoration: none; border-radius: 8px; font-weight: 600; }}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    rng = np.random.RandomState(42)
    n = 4000
    g1, g2, g3 = rng.randint(0, 31, n), rng.randint(0, 1428, n), rng.randint(0, 12568, n)
    haz, mag, soil, wet = rng.randint(1, 4, n), rng.uniform(4.5, 8.8, n), rng.randint(0, 3, n), rng.uniform(0, 1, n)
    age, area, ht, flr = rng.randint(0, 120, n), rng.randint(1, 50, n), rng.randint(2, 25, n), rng.randint(1, 8, n)
    fnd, roof = rng.randint(0, 4, n), rng.randint(0, 3, n)
    risk = (mag-5)*0.85 + (haz-1)*0.45 + (soil==2)*0.7 + wet*0.35 + (age>25)*0.6 + (flr>=3)*0.45 + (fnd==3)*1.1 + (roof==2)*0.7 + rng.normal(0, 0.35, n)
    y = np.where(risk < 2.2, 1, np.where(risk < 4.0, 2, 3))
    X = pd.DataFrame({"g1": g1, "g2": g2, "g3": g3, "haz": haz, "mag": mag, "soil": soil, "wet": wet, "age": age, "area": area, "ht": ht, "flr": flr, "fnd": fnd, "roof": roof})
    return RandomForestClassifier(n_estimators=40, random_state=42).fit(X, y)

@st.cache_data(ttl=1800)
def get_place(name):
    try:
        geo = json.loads(urllib.request.urlopen(f"https://open-meteo.com{urllib.parse.quote(name)}&count=1", timeout=5).read().decode())
        loc = next(iter(geo.get("results", [])), None)
        if not loc: return None
        lat, lon = float(loc["latitude"]), float(loc["longitude"])
        wx = json.loads(urllib.request.urlopen(f"https://open-meteo.com{lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation", timeout=5).read().decode()).get("current", {})
        eqs = json.loads(urllib.request.urlopen(f"https://usgs.gov{lat}&longitude={lon}&maxradiuskm=300&minmagnitude=4.0&limit=5", timeout=5).read().decode()).get("features", [])
        mx = max([float(f["properties"]["mag"]) for f in eqs], default=5.5)
        return {"name": f"{loc.get('name')}, {loc.get('country','')}", "lat": lat, "lon": lon, "temp": wx.get("temperature_2m", 25), "hum": wx.get("relative_humidity_2m", 60), "rain": wx.get("precipitation", 0), "eq": len(eqs), "max": round(mx, 1), "haz": 3 if mx >= 6.5 else 2}
    except Exception:
        return None

def bldg_html(flr, roof, fnd, age, grade, mode="m7", compact=False):
    col = "#64748b" if fnd == 0 else ("#94a3b8" if fnd == 1 else ("#b45309" if fnd == 2 else "#78350f"))
    crk = '<div style="position:absolute;width:24px;height:2px;background:#ef4444;transform:rotate(25deg);"></div>' if (age > 35 or grade >= 2) else ""
    floors_div = "".join([f'<div style="width:100px;height:17px;background:{col};border:1px solid #0f172a;display:flex;justify-content:space-around;align-items:center;position:relative;"><span style="width:12px;height:9px;background:#fef08a;"></span><span style="width:12px;height:9px;background:#fef08a;"></span>{crk}</div>' for _ in range(min(flr, 8))])
    roof_div = '<div style="width:108px;height:7px;background:#cbd5e1;"></div>' if roof == 0 else ('<div style="width:112px;height:9px;background:#38bdf8;"></div>' if roof == 1 else '<div style="width:0;height:0;border-left:55px solid transparent;border-right:55px solid transparent;border-bottom:18px solid #b91c1c;"></div>')
    btns = "" if compact else '<div style="margin-bottom:8px;"><button onclick="s(\'none\')">Normal</button> <button onclick="s(\'sh5 0.35s infinite\')">Mw 5.0</button> <button onclick="s(\'sh7 0.2s infinite\')">Mw 7.0</button> <button onclick="s(\'sh8 0.12s infinite\')">Mw 8.5</button></div>'
    init_a = "sh8 0.12s infinite" if mode == "m8" else ("sh7 0.2s infinite" if mode == "m7" else "sh5 0.35s infinite")
    return f'<div style="background:#0f172a;padding:12px;border-radius:12px;color:white;text-align:center;font-family:sans-serif;"><div style="color:#38bdf8;font-weight:bold;margin-bottom:6px;">🏗️ Custom Building Simulation ({flr} Floors | Grade {grade})</div>{btns}<div style="height:185px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;background:#1e293b;border-radius:8px;padding-bottom:6px;"><div id="b" style="display:flex;flex-direction:column;align-items:center;transform-origin:bottom;animation:{init_a};">{roof_div}{floors_div}</div><div style="width:80%;height:8px;background:#475569;"></div></div></div><style>@keyframes sh5{{0%,100%{{transform:translateX(0)}}50%{{transform:translateX(4px) rotate(0.6deg)}}}}@keyframes sh7{{0%,100%{{transform:translateX(0)}}50%{{transform:translateX(10px) rotate(2deg)}}}}@keyframes sh8{{0%,100%{{transform:translateX(0)}}50%{{transform:translateX(16px) rotate(4.5deg)}}}}</style><script>function s(a){{document.getElementById("b").style.animation=a;}}</script>'

@st.dialog("📤 Share Grade & Building Animation")
def share_popup(place, glbl, mag, flr, roof, fnd, age, grade, note):
    components.html(bldg_html(flr, roof, fnd, age, grade, "m8" if mag >= 7.5 else "m7", True), height=235)
    txt = f"🌍 GlobalQuake AI\n📍 {place} | {flr} Floors | Mw {mag}\n⚠️ {glbl}\n💬 {note}"
    st.code(txt)
    enc = urllib.parse.quote(txt)
    st.markdown(f'<a href="https://whatsapp.com{enc}" target="_blank">📲 Share on WhatsApp</a> | <a href="https://twitter.com{enc}" target="_blank">🐦 Share on X</a>', unsafe_allow_html=True)

model = load_model()
st.markdown('<div class="hero"><h1 style="margin:0;">🌍 GlobalQuake AI: Worldwide Damage Predictor</h1><p style="margin:6px 0 0;">Live Global Climate, Seismic Risk & Building Simulator</p></div>', unsafe_allow_html=True)

t1, t2 = st.tabs(["🔍 Worldwide Predictor & Simulator", "📊 Model Insights"])
with t1:
    st.markdown('<div class="tile">', unsafe_allow_html=True)
    place = st.text_input("🌐 Enter Any City or Country in the World", "Kathmandu, Nepal")
    p = get_place(place)
    if p:
        st.success(f"📍 {p['name']} ({p['lat']:.2f}, {p['lon']:.2f}) | 🌡️ {p['temp']}°C | 💧 {p['hum']}% | 🌧️ {p['rain']}mm | 🌋 Nearby Quakes: {p['eq']} (Max Mw {p['max']})")
        haz, wet, dmag = p["haz"], min(1.0, p["hum"]/100.0), max(5.5, min(8.8, p["max"]))
    else:
        haz, wet, dmag = 3, 0.5, 6.8
    c1, c2 = st.columns(2)
    with c1:
        g1 = st.number_input("Region ID (Geo 1)", 0, 30, 17)
        g2 = st.number_input("District ID (Geo 2)", 0, 1427, 450)
        g3 = st.number_input("Municipality ID (Geo 3)", 0, 12567, 3000)
        mag = st.slider("Magnitude (Mw)", 4.5, 9.0, float(dmag), 0.1)
        soil = {"Soft Alluvial Soil": 2, "Stiff Soil": 1, "Hard Bedrock": 0}[st.selectbox("Soil Type", ["Soft Alluvial Soil", "Stiff Soil", "Hard Bedrock"])]
    with c2:
        fnd = {"Mud Mortar - Stone/Brick": 3, "Bamboo/Timber": 2, "Cement/Brick": 1, "Reinforced Concrete (RCC)": 0}[st.selectbox("Foundation", ["Mud Mortar - Stone/Brick", "Bamboo/Timber", "Cement/Brick", "Reinforced Concrete (RCC)"])]
        roof = {"Heavy Timber/Mud": 2, "Light Metal Sheet": 1, "RCC Slab": 0}[st.selectbox("Roof Type", ["Heavy Timber/Mud", "Light Metal Sheet", "RCC Slab"])]
