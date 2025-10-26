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
    # 0) Exists?
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

    # 1) LFS pointer check
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            st.error(
                "⚠️ The file seems to be a Git LFS pointer, not the actual GeoJSON.\n\n"
                "Fix options:\n"
                "• Commit a simplified & gzipped file under 25MB without LFS (e.g., roads_wgs.geojson.gz), or\n"
                "• Host the file externally (Drive / GitHub Release) and download at runtime."
            )
            st.stop()
    except Exception:
        pass

    # 2) Read
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

    # 3) Disturbance + label
    gdf["disturbance"] = pd.to_numeric(gdf.get("disturbance"), errors="coerce")
    if "disturbance_label" not in gdf.columns:
        bins = [0, 0.33, 0.66, 1]
        gdf["disturbance_label"] = pd.cut(
            gdf["disturbance"], bins=bins,
            labels=["Low", "Medium", "High"],
            include_lowest=True
        )
    return gdf

# >>> set your file path here (use .gz if you committed gzipped)
DF_PATH = "data/roads_wgs.geojson.gz"
df = load_data(DF_PATH)
# If you committed plain GeoJSON:
# df = load_data("data/roads_wgs.geojson")

# =========================
# Clean geometries
# =========================
df = df[df.geometry.notna()].copy()
if hasattr(df.geometry, "is_empty"):
    df = df[~df.geometry.is_empty].copy()

# Keep only linear features
df = df[df.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

# Explode multilines
df = df.explode(index_parts=False, ignore_index=True)

# Disturbance → numeric [0,1]
df["disturbance"] = pd.to_numeric(df["disturbance"], errors="coerce").fillna(0.0).clip(0, 1)

# Label fallback
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
# Build GeoJSON manually (NumPy 2.x-safe)
# =========================
def to_py(obj):
    """JSON-serializable primitive converter for GeoJSON properties."""
    # numpy scalar → python scalar
    if isinstance(obj, np.generic):
        obj = obj.item()

    # datetime types → ISO
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # pandas interval, etc. → string
    if isinstance(obj, (pd.Interval,)):
        return str(obj)

    # arrays/lists → recursive
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_py(x) for x in obj]

    # NaN/NA safe check
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    # objects with tolist (e.g., pandas scalar types)
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
        feat = {
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {col: to_py(row[col]) for col in props_cols}
        }
        features.append(feat)
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
# Survey form (with 30s timeout + spinner + FR/EN normalize)
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
        # FR/EN normalize (in case someone types Oui/Non)
        agree_norm = {"Yes": "Yes", "No": "No", "Oui": "Yes", "Non": "No"}.get(agree, str(agree))

        payload = {
            "osmid": str(selected.get("osmid", "")),
            "highway": highway,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "agree": agree_norm,
            "rating_1to5": int(rating),
            "comment": comment,
            "click_lat": float(lat),
            "click_lon": float(lon),
            "user_lat": user_lat,
            "user_lon": user_lon
        }

        with st.spinner("Saving..."):
            try:
                r = requests.post(
                    APPSCRIPT_URL,
                    params={"token": APPSCRIPT_TOKEN},
                    json=payload,
                    timeout=30  # <-- artırılmış süre
                )
            except Exception as e:
                st.error(f"Connection error: {e}")
            else:
                if r.ok:
                    try:
                        resp = r.json()
                    except Exception:
                        resp = {"status": "?", "raw": r.text[:200]}
                    if resp.get("status") == "ok":
                        st.success("✅ Thanks! Your response has been saved.")
                        # İsterseniz kanıt bilgisi (Apps Script'ten dönerse) gösterebilirsiniz:
                        if resp.get("sheetName") and resp.get("wroteRow"):
                            st.caption(f"Sheet: {resp['sheetName']} • Row: {resp['wroteRow']}")
                    else:
                        st.error(f"Save failed: {resp}")
                else:
                    st.error(f"HTTP {r.status_code}: {r.text[:200]}")
else:
    st.info("Click on the map to select a segment.")
