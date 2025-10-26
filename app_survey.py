import os, json, gzip, requests
import numpy as np, pandas as pd, geopandas as gpd
import streamlit as st, folium
from shapely.geometry import Point, mapping
from streamlit_folium import st_folium
import branca.colormap as cm
from datetime import datetime

# =========================
# Page & language
# =========================
st.set_page_config(page_title="Urban Noise Survey", layout="wide")
LANG = st.radio("Language / Langue", ["English", "Français"], horizontal=True)

T = {
    "English": {
        "title": "🗺️ Urban Noise – Perception Survey",
        "intro": "Click on the map to select a road segment and submit your quick evaluation. That's it 👇",
        "legend": "Noise (low → high)",
        "no_lines": "No line geometry found.",
        "click_to_select": "Click on the map to select a segment.",
        "selected_sub": "Evaluate this segment",
        "road_model": "**Road:** {road}  |  **Model:** {label} (score={score:.2f})",
        "agree": "Do you agree with the model’s prediction?",
        "agree_opts": ["Yes", "No"],
        "rating": "Your perception (1 = very quiet, 5 = very noisy)",
        "comment": "Optional comment (traffic, construction, etc.)",
        "gps_exp": "Optional: share your GPS coordinates",
        "lat": "Latitude",
        "lon": "Longitude",
        "submit": "Submit",
        "saved": "✅ Thanks! Your response has been saved.",
        "save_failed": "Save failed",
        "http_err": "HTTP {code}: {text}",
        "conn_err": "Connection error: {e}",
        "missing_secret": "Missing secret: {k}\nManage app → Settings → Secrets\nAPPSCRIPT_URL, APPSCRIPT_TOKEN",
        "data_not_found": "Data file not found: {p}\nAvailable: {lst}",
        "geo_read_err": "❌ Geo read error: {e}",
    },
    "Français": {
        "title": "🗺️ Bruit urbain – Enquête de perception",
        "intro": "Cliquez sur la carte pour sélectionner un tronçon de route et envoyez votre évaluation rapide. C’est tout 👇",
        "legend": "Bruit (faible → élevé)",
        "no_lines": "Aucune géométrie linéaire trouvée.",
        "click_to_select": "Cliquez sur la carte pour sélectionner un tronçon.",
        "selected_sub": "Évaluer ce tronçon",
        "road_model": "**Route :** {road}  |  **Modèle :** {label} (score={score:.2f})",
        "agree": "Êtes-vous d’accord avec la prédiction du modèle ?",
        "agree_opts": ["Oui", "Non"],
        "rating": "Votre perception (1 = très calme, 5 = très bruyant)",
        "comment": "Commentaire facultatif (trafic, travaux, etc.)",
        "gps_exp": "Facultatif : partager vos coordonnées GPS",
        "lat": "Latitude",
        "lon": "Longitude",
        "submit": "Envoyer",
        "saved": "✅ Merci ! Votre réponse a été enregistrée.",
        "save_failed": "Échec de l’enregistrement",
        "http_err": "HTTP {code} : {text}",
        "conn_err": "Erreur de connexion : {e}",
        "missing_secret": "Secret manquant : {k}\nManage app → Settings → Secrets\nAPPSCRIPT_URL, APPSCRIPT_TOKEN",
        "data_not_found": "Fichier de données introuvable : {p}\nDisponibles : {lst}",
        "geo_read_err": "❌ Erreur de lecture Geo : {e}",
    },
}[LANG]

st.title(T["title"])
st.write(T["intro"])

# =========================
# Secrets (Apps Script)
# =========================
def require_secret(k: str) -> str:
    v = st.secrets.get(k)
    if not v:
        st.error(T["missing_secret"].format(k=k)); st.stop()
    return v

APPSCRIPT_URL   = require_secret("APPSCRIPT_URL")
APPSCRIPT_TOKEN = require_secret("APPSCRIPT_TOKEN")

# =========================
# Data loader (.geojson or .geojson.gz)
# =========================
@st.cache_data
def load_data(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        try: listing = os.listdir("data")
        except Exception: listing = []
        st.error(T["data_not_found"].format(p=path, lst=listing)); st.stop()

    # LFS pointer guard
    try:
        with open(path, "rb") as f:
            if f.read(64).startswith(b"version https://git-lfs.github.com/spec"):
                st.error("LFS pointer detected. Commit real .geojson(.gz) under 25MB."); st.stop()
    except Exception:
        pass

    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            gdf = gpd.GeoDataFrame.from_features(data["features"])
        else:
            gdf = gpd.read_file(path)
        gdf = gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)
    except Exception as e:
        st.error(T["geo_read_err"].format(e=e)); st.stop()

    gdf["disturbance"] = pd.to_numeric(gdf.get("disturbance"), errors="coerce").fillna(0.0).clip(0,1)
    if "disturbance_label" not in gdf.columns:
        gdf["disturbance_label"] = pd.cut(gdf["disturbance"], [0, .33, .66, 1],
                                          labels=["Low","Medium","High"], include_lowest=True)
    return gdf

DF_PATH = "data/roads_wgs.geojson.gz"
df = load_data(DF_PATH)

# =========================
# Clean geometries
# =========================
df = df[df.geometry.notna()].copy()
if hasattr(df.geometry, "is_empty"):
    df = df[~df.geometry.is_empty]
df = df[df.geometry.geom_type.isin(["LineString","MultiLineString"])].explode(index_parts=False, ignore_index=True)
if df.empty:
    st.error(T["no_lines"]); st.stop()

# =========================
# Helpers
# =========================
def to_py(o):
    if hasattr(o, "item"):
        try: o = o.item()
        except Exception: pass
    if isinstance(o, (pd.Timestamp, datetime)): return o.isoformat()
    try:
        if pd.isna(o): return None
    except Exception:
        pass
    return o

def build_geojson(gdf: gpd.GeoDataFrame) -> dict:
    props = [c for c in gdf.columns if c != gdf.geometry.name]
    feats = []
    for _, r in gdf.iterrows():
        g = r.geometry
        if g is None: continue
        feats.append({"type":"Feature","geometry":mapping(g),
                      "properties":{c: to_py(r[c]) for c in props}})
    return {"type":"FeatureCollection","features":feats}

# =========================
# Selection memory (for clear highlight)
# =========================
if "last_click" not in st.session_state:
    st.session_state.last_click = None

selected = None; sel_lat = sel_lon = None
if st.session_state.last_click:
    sel_lat, sel_lon = st.session_state.last_click
    clicked_pt = gpd.GeoSeries([Point(sel_lon, sel_lat)], crs=4326)
    idx = df.distance(clicked_pt.iloc[0]).sort_values().index[0]
    selected = df.loc[idx]

# =========================
# Map (vivid lines, semi-transparent)
# =========================
# More vivid ramp but not neon: green → yellow → orange → red
VIVID_RAMP = ['#00b050', '#ffff00', '#ff9900', '#ff0000']

vals = df["disturbance"].astype(float).clip(0,1)
q = np.quantile(vals, [0, .2, .4, .6, .8, 1])
def scaler(v): return (np.searchsorted(q, v, side="right")-1)/5

cmap = cm.LinearColormap(VIVID_RAMP, vmin=0, vmax=1)
cmap.caption = T["legend"]

center = [df.geometry.representative_point().y.mean(),
          df.geometry.representative_point().x.mean()]

# Slightly more colorful base than Positron
m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Voyager")

# Base colored lines — visible but see-through
folium.GeoJson(
    build_geojson(df),
    style_function=lambda f: {
        "color": cmap(float(scaler(float(f["properties"].get("disturbance",0))))),
        "weight": 5,
        "opacity": 0.75
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["highway","disturbance_label","disturbance"],
        aliases=["Road","Predicted","Score"] if LANG=="English" else ["Route","Prédiction","Score"]
    ),
    name="Noise lines"
).add_to(m)
cmap.add_to(m)

# Highlight selected segment (black) + click marker
if selected is not None:
    folium.GeoJson(
        build_geojson(selected.to_frame().T),
        style_function=lambda f: {"color":"#000000", "weight":7, "opacity":1.0},
        name="Selected"
    ).add_to(m)
    folium.CircleMarker(location=[sel_lat, sel_lon], radius=5, color="#000000",
                        fill=True, fill_opacity=1).add_to(m)

# Render & capture click
out = st_folium(m, height=600, use_container_width=True, returned_objects=["last_object_clicked"])

# New click → store & rerun to refresh highlight
if out and out.get("last_object_clicked"):
    lat = float(out["last_object_clicked"]["lat"])
    lon = float(out["last_object_clicked"]["lng"])
    st.session_state.last_click = (lat, lon)
    st.rerun()

# =========================
# Survey form & submit
# =========================
if selected is not None:
    st.subheader(T["selected_sub"])
    pred_label = str(selected.get("disturbance_label",""))
    pred_score = float(selected.get("disturbance",0.0))
    highway    = str(selected.get("highway",""))
    st.write(T["road_model"].format(road=highway, label=pred_label, score=pred_score))

    agree   = st.radio(T["agree"], T["agree_opts"], horizontal=True)
    rating  = st.slider(T["rating"], 1, 5, 3)
    comment = st.text_input(T["comment"])

    with st.expander(T["gps_exp"]):
        user_lat = st.text_input(T["lat"], "")
        user_lon = st.text_input(T["lon"], "")

    if st.button(T["submit"]):
        # Normalize FR Yes/No to English for consistent Sheets values
        agree_norm = agree
        if LANG == "Français":
            agree_norm = "Yes" if agree == "Oui" else "No"

        payload = {
            "osmid": str(selected.get("osmid","")),
            "highway": highway,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "agree": agree_norm,
            "rating_1to5": int(rating),
            "comment": comment,
            "click_lat": float(sel_lat),
            "click_lon": float(sel_lon),
            "user_lat": user_lat,
            "user_lon": user_lon
        }
        try:
            r = requests.post(APPSCRIPT_URL, params={"token": APPSCRIPT_TOKEN},
                              json=payload, timeout=10)
            if r.ok and r.headers.get("content-type","").startswith("application/json"):
                js = r.json()
                if js.get("status") == "ok":
                    st.success(T["saved"])
                else:
                    st.error(f'{T["save_failed"]}: {js}')
            else:
                st.error(T["http_err"].format(code=r.status_code, text=r.text[:200]))
        except Exception as e:
            st.error(T["conn_err"].format(e=e))
else:
    st.info(T["click_to_select"])
