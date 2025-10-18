import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from datetime import datetime
import requests
import os
import json
import gzip
from shapely.geometry import Point

st.set_page_config(page_title="Urban Noise Survey", layout="wide")
st.title("🗺️ Urban Noise – Perception Survey")

# ========== CONFIG (Secrets with guard) ==========
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

APPSCRIPT_URL = require_secret("APPSCRIPT_URL")      # "https://script.google.com/macros/s/XXX/exec"
APPSCRIPT_TOKEN = require_secret("APPSCRIPT_TOKEN")  # "YOUR_SHARED_TOKEN"


# ========== DATA LOADING (robust .geojson + .geojson.gz) ==========
@st.cache_data
def load_data(path: str):
    # 0) Dosya gerçekten var mı?
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

    # 1) Git LFS pointer mı? (ilk baytlara bak)
    # LFS pointer dosyaları düz metindir ve "version https://git-lfs.github.com/spec" ile başlar
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            st.error(
                "⚠️ The file seems to be a Git LFS pointer, not the actual GeoJSON.\n\n"
                "Fix options:\n"
                "• Commit a simplified & gzipped file under 25MB without LFS (e.g., roads_wgs.geojson.gz), or\n"
                "• Host the file externally (Google Drive / GitHub Release) and download at runtime."
            )
            st.stop()
    except Exception:
        pass

    # 2) Okuma: .gz ise manuel gzip aç; değilse read_file
    try:
        if path.endswith(".geojson.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            gdf = gpd.GeoDataFrame.from_features(data["features"])
            # CRS ayarı
            if gdf.crs is None:
                gdf.set_crs(4326, inplace=True)
            else:
                gdf = gdf.to_crs(4326)
        else:
            gdf = gpd.read_file(path).to_crs(4326)
    except Exception as e:
        st.error(f"❌ Failed to read geo data: {path}\n\n{e}")
        st.stop()

    # 3) Disturbance & linguistic label
    gdf["disturbance"] = pd.to_numeric(gdf.get("disturbance"), errors="coerce")
    if "disturbance_label" not in gdf.columns:
        bins = [0, 0.33, 0.66, 1]
        gdf["disturbance_label"] = pd.cut(
            gdf["disturbance"], bins=bins,
            labels=["Low", "Medium", "High"], include_lowest=True
        )
    return gdf


# >>> KENDİ DOSYA YOLUNA GÖRE BUNU KULLAN <<<
# Eğer repoya .gz yüklediysen:
df = load_data("data/roads_wgs.geojson.gz")
# Eğer ham .geojson yüklediysen (üst satırı yorumlayıp bunu aç):
# df = load_data("data/roads_wgs.geojson")


# ========== GDF TEMİZLİĞİ / STERİLİZASYON ==========
# 1) geometri yoksa / boşsa at
df = df[df.geometry.notna()].copy()
if "is_empty" in dir(df.geometry):
    df = df[~df.geometry.is_empty].copy()

# 2) yalnızca çizgisel geometriler kalsın
df = df[df.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

# 3) MultiLineString'leri satırlara böl
df = df.explode(index_parts=False, ignore_index=True)

# 4) disturbance sayıya dönsün ve NaN'leri güvenli bir değere çekelim
df["disturbance"] = pd.to_numeric(df["disturbance"], errors="coerce").fillna(0.0).clip(0, 1)

# 5) linguistic label yoksa üret
if "disturbance_label" not in df.columns:
    bins = [0, 0.33, 0.66, 1]
    df["disturbance_label"] = pd.cut(
        df["disturbance"], bins=bins,
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

# 6) Haritaya basmadan önce temel kontrol
if len(df) == 0:
    st.error("No line features to display after cleaning. Check your input data.")
    st.stop()


# ========== MAP ==========
center = [
    df.geometry.representative_point().y.mean(),
    df.geometry.representative_point().x.mean()
]
m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

cmap = cm.LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=0, vmax=1)
cmap.caption = "Disturbance Score (0–1)"

folium.GeoJson(
    df,
    style_function=lambda f: {
        "color": cmap(float(f["properties"].get("disturbance", 0))),
        "weight": 3, "opacity": 1.0
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

# ========== SELECTION ==========
selected = None
lat = lon = None
if out and out.get("last_object_clicked"):
    lat = float(out["last_object_clicked"]["lat"])
    lon = float(out["last_object_clicked"]["lng"])
    clicked_pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(df.crs)
    # En yakın segment
    idx = df.distance(clicked_pt.iloc[0]).sort_values().index[0]
    selected = df.loc[idx]

# ========== SURVEY FORM ==========
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
                params={"token": APPSCRIPT_TOKEN},  # basit doğrulama
                json=payload,
                timeout=10
            )
            if r.ok:
                # Apps Script JSON döndürmeli: {"status":"ok"}
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
