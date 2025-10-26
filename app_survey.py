import os
import json
import gzip
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
import uuid
from shapely.geometry import Point, mapping
from streamlit_folium import st_folium
import branca.colormap as cm
from datetime import datetime

st.set_page_config(page_title="Urban Noise Survey", layout="wide")

# =========================
# Language (EN / FR)
# =========================
LANG = st.radio("Language / Langue", ["English", "Français"], horizontal=True)

T = {
    "English": {
        "title": "🗺️ Urban Noise – Perception Survey",
        "caption": "Click a road segment to evaluate its perceived noise.",
        "legend": "Disturbance Score (0–1)",
        "click_info": "Click on the map to select a segment.",
        "subheader": "Evaluate this segment",
        "road_model": "**Road:** {road}  |  **Model prediction:** {label} (score={score:.2f})",
        "agree_q": "Do you agree with the prediction?",
        "agree_opts": ["Yes", "No"],
        "rating_q": "Your perception (1 = very quiet, 5 = very noisy)",
        "comment_q": "Optional comment (traffic, construction, etc.)",
        "gps_exp": "Optional: share your GPS coordinates",
        "lat": "Latitude",
        "lon": "Longitude",
        "submit": "Submit",
        "thanks": "✅ Thanks! Your response has been saved.",
        "save_failed": "Save failed",
        "conn_err": "Connection error: {e}",
    },
    "Français": {
        "title": "🗺️ Bruit urbain – Enquête de perception",
        "caption": "Cliquez sur un tronçon de route pour évaluer le bruit perçu.",
        "legend": "Indice de nuisance (0–1)",
        "click_info": "Cliquez sur la carte pour sélectionner un tronçon.",
        "subheader": "Évaluer ce tronçon",
        "road_model": "**Route :** {road}  |  **Prédiction du modèle :** {label} (score={score:.2f})",
        "agree_q": "Êtes-vous d’accord avec la prédiction ?",
        "agree_opts": ["Oui", "Non"],
        "rating_q": "Votre perception (1 = très calme, 5 = très bruyant)",
        "comment_q": "Commentaire facultatif (trafic, travaux, etc.)",
        "gps_exp": "Facultatif : partager vos coordonnées GPS",
        "lat": "Latitude",
        "lon": "Longitude",
        "submit": "Envoyer",
        "thanks": "✅ Merci ! Votre réponse a été enregistrée.",
        "save_failed": "Échec de l’enregistrement",
        "conn_err": "Erreur de connexion : {e}",
    }
}[LANG]

st.title(T["title"])

# =========================
# Secrets guard
# =========================
def require_secret(key_name: str) -> str:
    val = st.secrets.get(key_name)
    if not val:
        st.error(f"Missing secret: {key_name}")
        st.stop()
    return val

APPSCRIPT_URL = require_secret("APPSCRIPT_URL")
APPSCRIPT_TOKEN = require_secret("APPSCRIPT_TOKEN")

# =========================
# Load GeoData
# =========================
@st.cache_data
def load_data(path: str) -> gpd.GeoDataFrame:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        gdf = gpd.GeoDataFrame.from_features(data["features"])
    else:
        gdf = gpd.read_file(path)
    gdf = gdf.to_crs(4326)
    gdf["disturbance"] = pd.to_numeric(gdf.get("disturbance"), errors="coerce").fillna(0)
    if "disturbance_label" not in gdf.columns:
        gdf["disturbance_label"] = pd.cut(
            gdf["disturbance"], [0, 0.33, 0.66, 1],
            labels=["Low", "Medium", "High"], include_lowest=True
        )
    return gdf

DF_PATH = "data/roads_wgs.geojson.gz"
df = load_data(DF_PATH)

# =========================
# Map
# =========================
center = [df.geometry.representative_point().y.mean(),
          df.geometry.representative_point().x.mean()]
m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")
cmap = cm.LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=0, vmax=1)
cmap.caption = T["legend"]

folium.GeoJson(
    df,
    style_function=lambda f: {
        "color": cmap(float(f["properties"].get("disturbance", 0))),
        "weight": 3,
        "opacity": 1.0,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["highway", "disturbance_label", "disturbance"],
        aliases=(["Road", "Predicted Level", "Score"]
                 if LANG == "English" else ["Route", "Niveau prédit", "Score"])
    )
).add_to(m)
cmap.add_to(m)

st.caption(T["caption"])
out = st_folium(m, height=600, use_container_width=True, returned_objects=["last_object_clicked"])

# =========================
# Handle Click
# =========================
selected = None
lat = lon = None
if out and out.get("last_object_clicked"):
    lat = float(out["last_object_clicked"]["lat"])
    lon = float(out["last_object_clicked"]["lng"])
    clicked_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326)
    idx = df.distance(clicked_pt.iloc[0]).sort_values().index[0]
    selected = df.loc[idx]

# =========================
# Survey
# =========================
if selected is not None:
    st.subheader(T["subheader"])
    pred_label = str(selected.get('disturbance_label', ''))
    pred_score = float(selected.get('disturbance', 0.0))
    highway = str(selected.get('highway', ''))

    st.write(T["road_model"].format(road=highway, label=pred_label, score=pred_score))
    agree = st.radio(T["agree_q"], T["agree_opts"], horizontal=True)
    rating = st.slider(T["rating_q"], 1, 5, 3)
    comment = st.text_input(T["comment_q"])
    with st.expander(T["gps_exp"]):
        user_lat = st.text_input(T["lat"], "")
        user_lon = st.text_input(T["lon"], "")

    if st.button(T["submit"]):
        req_uuid = str(uuid.uuid4())
        agree_norm = {"Yes": "Yes", "No": "No", "Oui": "Yes", "Non": "No"}.get(agree, agree)
        payload = {
            "uuid": req_uuid,
            "osmid": str(selected.get("osmid", "")),
            "highway": highway,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "agree": agree_norm,
            "rating_1to5": int(rating),
            "comment": comment,
            "click_lat": lat,
            "click_lon": lon,
            "user_lat": user_lat,
            "user_lon": user_lon,
        }

        with st.spinner("Saving..."):
            try:
                r = requests.post(
                    APPSCRIPT_URL,
                    params={"token": APPSCRIPT_TOKEN},
                    json=payload,
                    timeout=30,  # ⏱️ Artırılmış süre
                )
            except Exception as e:
                st.error(T["conn_err"].format(e=e))
            else:
                if r.ok:
                    try:
                        resp = r.json()
                    except Exception:
                        resp = {"status": "?", "raw": r.text[:200]}
                    if resp.get("status") == "ok":
                        st.success(T["thanks"])
                        st.caption(
                            f"Sheet: {resp.get('sheetName')} | Row: {resp.get('wroteRow')} | ID: {resp.get('uuid')}\n\n"
                            f"{resp.get('spreadsheetUrl')}"
                        )
                    else:
                        st.error(f"{T['save_failed']}: {resp}")
                else:
                    st.error(f"HTTP {r.status_code}: {r.text[:200]}")
else:
    st.info(T["click_info"])
