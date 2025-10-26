import os
import json
import gzip
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from shapely.geometry import Point, mapping
from streamlit_folium import st_folium
import branca.colormap as cm
from datetime import datetime

st.set_page_config(page_title="Urban Noise Survey", layout="wide")
st.title("🗺️ Urban Noise – Perception Survey")

# =========================
#  Google Form ayarları
# =========================
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeVqxIIMNZ4j3hvz6owNyD5C7g9KLr6y7vccZYCetImk0XHhA/formResponse"

ENTRY_MAP = {
    "osmid":       "entry.131606148",
    "highway":     "entry.425157021",
    "pred_label":  "entry.264773928",
    "pred_score":  "entry.108563380",
    "agree":       "entry.434809201",   # seçenek metni tam "Yes"/"No" olmalı
    "rating_1to5": "entry.558617464",
    "comment":     "entry.1491013630",
    "click_lat":   "entry.873151532",
    "click_lon":   "entry.1998137056",
    "user_lat":    "entry.1204200612",
    "user_lon":    "entry.1041474888",
}

def send_to_google_form(payload: dict, timeout: int = 20):
    """Google Form'a POST atar. 200 veya 302 başarılı kabul edilir."""
    data = {
        ENTRY_MAP["osmid"]:       str(payload.get("osmid","")),
        ENTRY_MAP["highway"]:     str(payload.get("highway","")),
        ENTRY_MAP["pred_label"]:  str(payload.get("pred_label","")),
        ENTRY_MAP["pred_score"]:  str(payload.get("pred_score","")),
        ENTRY_MAP["agree"]:       str(payload.get("agree","")),
        ENTRY_MAP["rating_1to5"]: str(payload.get("rating_1to5","")),
        ENTRY_MAP["comment"]:     str(payload.get("comment","")),
        ENTRY_MAP["click_lat"]:   str(payload.get("click_lat","")),
        ENTRY_MAP["click_lon"]:   str(payload.get("click_lon","")),
        ENTRY_MAP["user_lat"]:    str(payload.get("user_lat","")),
        ENTRY_MAP["user_lon"]:    str(payload.get("user_lon","")),
    }
    r = requests.post(FORM_URL, data=data, timeout=timeout)
    return r.status_code in (200, 302), r.status_code, r.text[:200]

# =========================
# Data loader (robust .geojson + .geojson.gz)
# =========================
@st.cache_data
def load_data(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        try:
            listing = os.listdir("data")
        except Exception:
            listing = []
        st.error(
            f"Data file not found: {path}\n\n"
            f"Available in ./data: {listing}\n"
            f"Tip: If you committed roads_wgs.geojson.gz, call load_data('data/roads_wgs.geojson.gz')."
        )
        st.stop()

    # LFS pointer kontrolü
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            st.error(
                "⚠️ The file seems to be a Git LFS pointer, not the actual GeoJSON.\n\n"
                "Fix: commit a simplified gzipped file under 25MB (roads_wgs.geojson.gz)."
            )
            st.stop()
    except Exception:
        pass

    try:
        if path.endswith(".geojson.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            gdf = gpd.GeoDataFrame.from_features(data["features"])
            if gdf.crs is None:
                gdf.set_crs(4326, inplace=True)
            else:
                gdf = gdf.to_crs(4326)
        else:
            gdf = gpd.read_file(path).to_crs(4326)
    except Exception as e:
        st.error(f"❌ Failed to read geo data: {path}\n\n{e}")
        st.stop()

    # disturbance + label
    gdf["disturbance"] = pd.to_numeric(gdf.get("disturbance"), errors="coerce")
    if "disturbance_label" not in gdf.columns:
        bins = [0, 0.33, 0.66, 1]
        gdf["disturbance_label"] = pd.cut(
            gdf["disturbance"], bins=bins,
            labels=["Low", "Medium", "High"],
            include_lowest=True
        )
    return gdf

DF_PATH = "data/roads_wgs.geojson.gz"
df = load_data(DF_PATH)

# =========================
# Clean geometries
# =========================
df = df[df.geometry.notna()].copy()
if hasattr(df.geometry, "is_empty"):
    df = df[~df.geometry.is_empty].copy()
df = df[df.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
df = df.explode(index_parts=False, ignore_index=True)
df["disturbance"] = pd.to_numeric(df["disturbance"], errors="coerce").fillna(0.0).clip(0, 1)
if "disturbance_label" not in df.columns:
    bins = [0, 0.33, 0.66, 1]
    df["disturbance_label"] = pd.cut(df["disturbance"], bins=bins,
                                     labels=["Low", "Medium", "High"], include_lowest=True)
if len(df) == 0:
    st.error("No line features to display after cleaning. Check your input data.")
    st.stop()

# =========================
# Build GeoJSON manually
# =========================
def to_py(obj):
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj

def build_geojson(gdf: gpd.GeoDataFrame) -> dict:
    props_cols = [c for c in gdf.columns if c != gdf.geometry.name]
    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {col: to_py(row[col]) for col in props_cols}
        })
    return {"type": "FeatureCollection", "features": features}

geojson_data = build_geojson(df)

# =========================
# Map
# =========================
center = [
    df.geometry.representative_point().y.mean(),
    df.geometry.representative_point().x.mean()
]
m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

cmap = cm.LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=0, vmax=1)
cmap.caption = "Disturbance Score (0–1)"

folium.GeoJson(
    geojson_data,
    style_function=lambda f: {
        "color": cmap(float(f["properties"].get("disturbance", 0))),
        "weight": 3,
        "opacity": 1.0
    },
    highlight_function=lambda f: {"weight": 6},
    tooltip=folium.GeoJsonTooltip(
        fields=["highway", "disturbance_label", "disturbance"],
        aliases=["Road", "Predicted Level", "Score"]
    ),
    name="Roads"
).add_to(m)
cmap.add_to(m)

st.caption("Click a road segment to evaluate its perceived noise.")
out = st_folium(m, height=600, use_container_width=True, returned_objects=["last_object_clicked"])

# =========================
# Selection (nearest segment to click)
# =========================
selected = None
lat = lon = None
if out and out.get("last_object_clicked"):
    lat = float(out["last_object_clicked"]["lat"])
    lon = float(out["last_object_clicked"]["lng"])
    clicked_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(df.crs)
    idx = df.distance(clicked_pt.iloc[0]).sort_values().index[0]
    selected = df.loc[idx]

# =========================
# Survey form -> Google Form'a gönder
# =========================
if selected is not None:
    st.subheader("Evaluate this segment")
    pred_label = str(selected.get('disturbance_label', ''))
    pred_score = float(selected.get('disturbance', 0.0))
    highway = str(selected.get('highway', ''))

    st.write(f"**Road:** {highway}  |  **Model prediction:** {pred_label} (score={pred_score:.2f})")

    agree = st.radio("Do you agree with the prediction?", ["Yes", "No"], horizontal=True)
    rating = st.slider("Your perception (1 = very quiet, 5 = very noisy)", 1, 5, 3)
    comment = st.text_input("Optional comment (traffic, construction, etc.)")
    with st.expander("Optional: share your GPS coordinates"):
        user_lat = st.text_input("Latitude", "")
        user_lon = st.text_input("Longitude", "")

    if st.button("Submit"):
        payload = {
            "osmid": str(selected.get("osmid", "")),
            "highway": highway,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "agree": agree,                # Formdaki seçenek metniyle aynı olmalı
            "rating_1to5": int(rating),
            "comment": comment,
            "click_lat": float(lat),
            "click_lon": float(lon),
            "user_lat": user_lat,
            "user_lon": user_lon
        }
        with st.spinner("Sending to Google Form..."):
            ok, code, preview = send_to_google_form(payload, timeout=20)
        if ok:
            st.success("✅ Submitted! Check the Form responses (and linked Sheet).")
        else:
            st.warning(f"Response code: {code}. Check FORM_URL & ENTRY_MAP.")
else:
    st.info("Click on the map to select a segment.")
