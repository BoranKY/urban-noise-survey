import os
import json
import gzip
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from folium import plugins  # HeatMap, Fullscreen, MeasureControl
from shapely.geometry import Point, mapping
from streamlit_folium import st_folium
import branca.colormap as cm
from datetime import datetime

st.set_page_config(page_title="Urban Noise Survey", layout="wide")
st.title("🗺️ Urban Noise – Perception Survey")

# =========================
# Secrets guard
# =========================
def require_secret(key_name: str) -> str:
    val = st.secrets.get(key_name)
    if not val:
        st.error(
            f"Missing secret: {key_name}\n\n"
            "Go to: Manage app → Settings → Secrets and add:\n"
            'APPSCRIPT_URL = "https://script.google.com/macros/s/.../exec"\n'
            'APPSCRIPT_TOKEN = "YOUR_SHARED_TOKEN"'
        )
        st.stop()
    return val

APPSCRIPT_URL = require_secret("APPSCRIPT_URL")
APPSCRIPT_TOKEN = require_secret("APPSCRIPT_TOKEN")

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

    # LFS pointer check
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            st.error(
                "⚠️ The file seems to be a Git LFS pointer, not the actual GeoJSON.\n\n"
                "Fix options:\n"
                "• Commit a simplified & gzipped file under 25MB without LFS (e.g., roads_wgs.geojson.gz), or\n"
                "• Host the file externally and download at runtime."
            )
            st.stop()
    except Exception:
        pass

    # Read
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

    # Disturbance + label
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
    df["disturbance_label"] = pd.cut(
        df["disturbance"], bins=bins,
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

if len(df) == 0:
    st.error("No line features to display after cleaning. Check your input data.")
    st.stop()

# =========================
# Helpers
# =========================
def to_py(obj):
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (pd.Interval,)):
        return str(obj)
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_py(x) for x in obj]
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

# =========================
# Map (configurable heat view)
# =========================
with st.sidebar:
    st.header("🗺️ Map settings")
    map_zoom = st.slider("Zoom start", 8, 18, 13)
    line_weight = st.slider("Line weight", 1, 10, 4)
    line_opacity = st.slider("Line opacity", 0.1, 1.0, 0.9)
    cmap_name = st.selectbox("Color ramp", ["green→yellow→red", "blue→purple→red"])
    scheme = st.selectbox("Classification", ["continuous", "quantile (q=5)", "equal bins (k=5)"])
    show_line = st.checkbox("Show colored lines", True)
    show_heat = st.checkbox("Show point heatmap", True)
    min_score = st.slider("Min disturbance to show", 0.0, 1.0, 0.0, 0.01)
    highway_filter = st.multiselect(
        "Filter by highway", sorted(df["highway"].dropna().astype(str).unique().tolist()), []
    )

vis = df.copy()
if highway_filter:
    vis = vis[vis["highway"].astype(str).isin(highway_filter)]
vis = vis[vis["disturbance"] >= float(min_score)]
if vis.empty:
    st.warning("No features match current filters.")
    st.stop()

center = [
    vis.geometry.representative_point().y.mean(),
    vis.geometry.representative_point().x.mean()
]
m = folium.Map(location=center, zoom_start=map_zoom, tiles="cartodbpositron")

# Colormap
ramp = ['#2c7fb8', '#7b3294', '#d7191c'] if cmap_name == "blue→purple→red" else ['green', 'yellow', 'red']

vals = vis["disturbance"].astype(float).clip(0, 1)
if scheme.startswith("quantile"):
    q = np.quantile(vals, [0, .2, .4, .6, .8, 1])
    def scaler(v): return (np.searchsorted(q, v, side="right")-1)/5
elif scheme.startswith("equal"):
    bins = np.linspace(vals.min(), vals.max(), 6)
    def scaler(v): return (np.searchsorted(bins, v, side="right")-1)/5
else:
    vmin, vmax = float(vals.min()), float(vals.max())
    span = max(vmax - vmin, 1e-9)
    def scaler(v): return (float(v) - vmin) / span

cmap = cm.LinearColormap(ramp, vmin=0, vmax=1)
cmap.caption = "Disturbance (scaled)"

# LAYER 1: colored line choropleth
if show_line:
    folium.GeoJson(
        build_geojson(vis),
        style_function=lambda f: {
            "color": cmap(float(scaler(f["properties"].get("disturbance", 0)))),
            "weight": line_weight,
            "opacity": line_opacity
        },
        highlight_function=lambda f: {"weight": line_weight + 2},
        tooltip=folium.GeoJsonTooltip(
            fields=["highway", "disturbance_label", "disturbance"],
            aliases=["Road", "Predicted Level", "Score"]
        ),
        name="Line heat"
    ).add_to(m)
    cmap.add_to(m)

# LAYER 2: weighted point heatmap
if show_heat:
    reps = vis.copy()
    reps["pt"] = reps.geometry.representative_point()
    heat_data = [
        [geom.y, geom.x, float(w)]
        for geom, w in zip(reps["pt"], reps["disturbance"])
        if geom is not None
    ]
    if heat_data:
        plugins.HeatMap(
            heat_data,
            name="Point heatmap",
            radius=18,
            blur=15,
            max_zoom=18,
            min_opacity=0.3
        ).add_to(m)

plugins.Fullscreen(position="topleft").add_to(m)
plugins.MeasureControl(primary_length_unit='meters').add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st.caption("Tip: Use the sidebar to change classification, filters and layers.")
out = st_folium(m, height=650, use_container_width=True, returned_objects=["last_object_clicked"])

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
# Survey form & submit
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
            "agree": agree,
            "rating_1to5": int(rating),
            "comment": comment,
            "click_lat": float(lat),
            "click_lon": float(lon),
            "user_lat": user_lat,
            "user_lon": user_lon
        }
        try:
            r = requests.post(
                APPSCRIPT_URL,
                params={"token": APPSCRIPT_TOKEN},
                json=payload,
                timeout=10
            )
            if r.ok:
                try:
                    resp = r.json()
                except Exception:
                    resp = {"status": "?", "raw": r.text[:200]}
                if resp.get("status") == "ok":
                    st.success("✅ Thanks! Your response has been saved.")
                else:
                    st.error(f"Save failed: {resp}")
            else:
                st.error(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            st.error(f"Connection error: {e}")
else:
    st.info("Click on the map to select a segment.")
