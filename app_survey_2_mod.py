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
from datetime import datetime

# (Optional) get user geolocation; if missing, we fall back to blanks
try:
    from streamlit_js_eval import get_geolocation
    HAS_JS_GEO = True
except Exception:
    HAS_JS_GEO = False

# =========================
#  Page & Header
# =========================
st.set_page_config(page_title="How Noisy Is This Street?", layout="wide")
st.title("How Noisy Is This Street?")

st.markdown("""
**Help us map where streets feel quiet or noisy.** Your feedback improves walking and cycling routes.

**Colors on the map:**
- 🟢 **Green** = Low (quieter)
- 🟡 **Yellow** = Medium
- 🔴 **Red** = High (noisier)

**How it works (takes ~1 minute):**
- **Zoom** to your area on the map.  
- **Click** a street you know.  
- **Rate how much you agree with the map’s noise prediction:**  
  **1 = don’t agree at all … 5 = completely agree.**  
- (Optional) **Add a short note** (e.g., “road works”, “rush hour”).  
- **Submit** your answer.

**Important:** Loading the map can take a bit the first time.  
After each **click** or **submit**, the page may take a few seconds to process.  
Please **stay on this page** until you see the confirmation message.

**Privacy:** We only store your answer and the clicked map location — no name or email.
""")

# =========================
#  Google Form settings
# =========================
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeVqxIIMNZ4j3hvz6owNyD5C7g9KLr6y7vccZYCetImk0XHhA/formResponse"

ENTRY_MAP = {
    "osmid":       "entry.131606148",
    "highway":     "entry.425157021",
    "pred_label":  "entry.264773928",
    "pred_score":  "entry.108563380",
    "agree":       "entry.434809201",   # we will send "NaN"
    "rating_1to5": "entry.558617464",
    "comment":     "entry.1491013630",
    "click_lat":   "entry.873151532",
    "click_lon":   "entry.1998137056",
    "user_lat":    "entry.1204200612",
    "user_lon":    "entry.1041474888",
}

def send_to_google_form(payload: dict, timeout: int = 20):
    """POST to Google Form. 200 or 302 considered success."""
    data = {
        ENTRY_MAP["osmid"]:       str(payload.get("osmid","")),
        ENTRY_MAP["highway"]:     str(payload.get("highway","")),
        ENTRY_MAP["pred_label"]:  str(payload.get("pred_label","")),
        ENTRY_MAP["pred_score"]:  str(payload.get("pred_score","")),
        ENTRY_MAP["agree"]:       str(payload.get("agree","")),  # "NaN"
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
#  Data loader (.geojson / .geojson.gz)
#  + geometry cleaning (cached)
# =========================
@st.cache_data
def load_data(path: str) -> gpd.GeoDataFrame:
    """Load and clean the roads GeoDataFrame (runs once, then cached)."""
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
                "Fix: commit a simplified gzipped file under 25MB (roads_wgs.geojson.gz)."
            )
            st.stop()
    except Exception:
        pass

    # Read file
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

    # --- Geometry cleaning (moved buraya, bir kez çalışsın diye) ---
    gdf = gdf[gdf.geometry.notna()].copy()
    if hasattr(gdf.geometry, "is_empty"):
        gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.explode(index_parts=False, ignore_index=True)

    # Disturbance numeric + label
    gdf["disturbance"] = pd.to_numeric(gdf.get("disturbance"), errors="coerce").fillna(0.0).clip(0, 1)
    if "disturbance_label" not in gdf.columns:
        bins = [0, 0.33, 0.66, 1]
        gdf["disturbance_label"] = pd.cut(
            gdf["disturbance"], bins=bins,
            labels=["Low", "Medium", "High"],
            include_lowest=True
        )

    if len(gdf) == 0:
        st.error("No line features to display after cleaning. Check your input data.")
        st.stop()

    return gdf

# Helper for json conversion
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

@st.cache_data
def get_geojson_data(path: str) -> dict:
    """Build GeoJSON dict for Folium from the cached GeoDataFrame."""
    gdf = load_data(path)
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

# Spatial index (nearest search) - cache as resource
@st.cache_resource
def get_sindex(_gdf: gpd.GeoDataFrame):
    """Create a spatial index for fast nearest-segment queries.

    _gdf is prefixed with underscore so Streamlit doesn't try to hash it.
    """
    try:
        return _gdf.sindex
    except Exception:
        return None

# =========================
#  Load data (cached)
# =========================
DF_PATH = "data/roads_wgs.geojson.gz"  # gerekirse burada dosya adını değiştir
df = load_data(DF_PATH)
geojson_data = get_geojson_data(DF_PATH)
sindex = get_sindex(df)

# =========================
# Map (no legend, no tooltip)
# =========================
center = [
    df.geometry.representative_point().y.mean(),
    df.geometry.representative_point().x.mean()
]
m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

from branca.colormap import LinearColormap
cmap = LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=0, vmax=1)

folium.GeoJson(
    geojson_data,
    style_function=lambda f: {
        "color": cmap(float(f["properties"].get("disturbance", 0))),
        "weight": 3 if float(f["properties"].get("disturbance", 0)) < 0.7 else 4,
        "opacity": 0.9 if float(f["properties"].get("disturbance", 0)) >= 0.33 else 0.6
    },
    highlight_function=lambda f: {"weight": 6},
    name="Roads"
).add_to(m)

st.caption("Click a street line, then use the sidebar to submit your feedback. (Processing may take a few seconds.)")
out = st_folium(m, height=600, use_container_width=True, returned_objects=["last_object_clicked"])

# =========================
# Selection (nearest segment) + optional geolocation
# =========================
selected = None
lat = lon = None

if out and out.get("last_object_clicked"):
    lat = float(out["last_object_clicked"]["lat"])
    lon = float(out["last_object_clicked"]["lng"])
    click_geom = Point(lon, lat)  # df already in 4326

    # Prefer spatial index (fast), fall back to distance if something goes wrong
    if sindex is not None:
        try:
            # shapely/pygeos backend: returns indices
            nearest_idx = sindex.nearest(click_geom, return_all=False)

            # defensive: handle different shapes/backends
            tree_idx = None
            if hasattr(nearest_idx, "shape") and nearest_idx.shape[0] == 2:
                tree_idx = int(nearest_idx[1][0])
            else:
                # rtree-style or 1D array / iterable
                idx_list = list(nearest_idx)
                tree_idx = int(idx_list[0])

            selected = df.iloc[tree_idx]
        except Exception:
            # fallback: brute-force distance (slower)
            distances = df.distance(click_geom)
            idx = distances.sort_values().index[0]
            selected = df.loc[idx]
    else:
        distances = df.distance(click_geom)
        idx = distances.sort_values().index[0]
        selected = df.loc[idx]

user_lat = user_lon = ""
if HAS_JS_GEO:
    try:
        loc = get_geolocation()
        if loc and loc.get("coords"):
            user_lat = str(loc["coords"]["latitude"])
            user_lon = str(loc["coords"]["longitude"])
    except Exception:
        pass

# =========================
# Sidebar Form (rating slider + optional comment)
# =========================
st.sidebar.header("Rate this street")

if selected is None:
    st.sidebar.info("Click a street on the map to start.")
else:
    pred_label = str(selected.get('disturbance_label', ''))
    pred_score = float(selected.get('disturbance', 0.0))
    highway = str(selected.get('highway', ''))

    rating = st.sidebar.slider(
        "How much do you agree with this street’s noise prediction? "
        "(1 = don’t agree at all, 5 = completely agree)",
        1, 5, 3, key="rating_agree"
    )

    if rating <= 2:
        st.sidebar.warning(
            "You seem to disagree. Please tell us why, and which color you think "
            "this street should be (green / yellow / red)."
        )

    comment = st.sidebar.text_input("Optional comment (e.g., road works, rush hour)", key="comment")

    if st.sidebar.button("Submit", use_container_width=True):
        payload = {
            "osmid": str(selected.get("osmid", "")),
            "highway": highway,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "agree": "NaN",             # UI removed; send 'NaN' string
            "rating_1to5": int(rating),
            "comment": comment,
            "click_lat": float(lat) if lat is not None else "",
            "click_lon": float(lon) if lon is not None else "",
            "user_lat": user_lat,
            "user_lon": user_lon
        }
        with st.spinner("Submitting… please wait a few seconds"):
            ok, code, preview = send_to_google_form(payload, timeout=20)
        if ok:
            st.sidebar.success("✅ Thanks! Your response has been saved.")
        else:
            st.sidebar.warning(f"Submission issue (HTTP {code}). Please try again.")
