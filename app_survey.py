import os, json, gzip, requests
import numpy as np, pandas as pd, geopandas as gpd
import streamlit as st, folium
from shapely.geometry import Point, mapping
from streamlit_folium import st_folium
import branca.colormap as cm
from datetime import datetime

st.set_page_config(page_title="Urban Noise Survey", layout="wide")
st.title("🗺️ Urban Noise – Perception Survey")
st.write("Haritaya tıklayıp bir yol parçası seçin ve kısa değerlendirmeyi gönderin. Hepsi bu 👇")

# ---------- secrets ----------
def require_secret(k: str) -> str:
    v = st.secrets.get(k)
    if not v:
        st.error(
            f"Missing secret: {k}\n"
            "Manage app → Settings → Secrets içine ekleyin:\n"
            'APPSCRIPT_URL="https://script.google.com/macros/s/.../exec"\n'
            'APPSCRIPT_TOKEN="YOUR_SHARED_TOKEN"'
        ); st.stop()
    return v

APPSCRIPT_URL  = require_secret("APPSCRIPT_URL")
APPSCRIPT_TOKEN = require_secret("APPSCRIPT_TOKEN")

# ---------- data loader ----------
@st.cache_data
def load_data(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        try: listing = os.listdir("data")
        except Exception: listing = []
        st.error(f"Data file not found: {path}\nAvailable: {listing}"); st.stop()

    # LFS pointer koruması
    try:
        with open(path, "rb") as f:
            if f.read(64).startswith(b"version https://git-lfs.github.com/spec"):
                st.error("Bu dosya LFS pointer gibi görünüyor. Gerçek .geojson(.gz) dosyasını commit edin."); st.stop()
    except Exception: pass

    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            gdf = gpd.GeoDataFrame.from_features(data["features"])
        else:
            gdf = gpd.read_file(path)
        gdf = gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)
    except Exception as e:
        st.error(f"❌ Geo okuma hatası: {e}"); st.stop()

    gdf["disturbance"] = pd.to_numeric(gdf.get("disturbance"), errors="coerce").fillna(0.0).clip(0,1)
    if "disturbance_label" not in gdf.columns:
        gdf["disturbance_label"] = pd.cut(gdf["disturbance"], [0, .33, .66, 1],
                                          labels=["Low","Medium","High"], include_lowest=True)
    return gdf

DF_PATH = "data/roads_wgs.geojson.gz"
df = load_data(DF_PATH)

# ---------- clean ----------
df = df[df.geometry.notna()].copy()
if hasattr(df.geometry, "is_empty"): df = df[~df.geometry.is_empty]
df = df[df.geometry.geom_type.isin(["LineString","MultiLineString"])].explode(index_parts=False, ignore_index=True)
if df.empty: st.error("Çizgi geometri bulunamadı."); st.stop()

# ---------- helpers ----------
def to_py(o):
    if hasattr(o, "item"): 
        try: o = o.item()
        except Exception: pass
    if isinstance(o, (pd.Timestamp, datetime)): return o.isoformat()
    try:
        if pd.isna(o): return None
    except Exception: pass
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

# ---------- remember last click (seçimi vurgulamak için) ----------
if "last_click" not in st.session_state: st.session_state.last_click = None

# ---------- selection from stored click (bir önceki tıklamayı vurgula) ----------
selected = None; sel_lat = sel_lon = None
if st.session_state.last_click:
    sel_lat, sel_lon = st.session_state.last_click
    clicked_pt = gpd.GeoSeries([Point(sel_lon, sel_lat)], crs=4326)
    idx = df.distance(clicked_pt.iloc[0]).sort_values().index[0]
    selected = df.loc[idx]

# ---------- map (basit & otomatik sınıflama) ----------
# quantile (q=5) sınıflama ile daha dengeli renkler
vals = df["disturbance"].astype(float).clip(0,1)
q = np.quantile(vals, [0, .2, .4, .6, .8, 1])
def scaler(v): return (np.searchsorted(q, v, side="right")-1)/5
cmap = cm.LinearColormap(['green','yellow','red'], vmin=0, vmax=1)
cmap.caption = "Noise (low → high)"

center = [df.geometry.representative_point().y.mean(),
          df.geometry.representative_point().x.mean()]
m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

# temel katman
folium.GeoJson(
    build_geojson(df),
    style_function=lambda f: {
        "color": cmap(float(scaler(float(f["properties"].get("disturbance",0))))),
        "weight": 5, "opacity": 0.95
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["highway","disturbance_label","disturbance"],
        aliases=["Road","Predicted","Score"]
    ),
    name="Noise lines"
).add_to(m)
cmap.add_to(m)

# seçili segmenti kalın çiz ve tıklama noktasına küçük işaret koy
if selected is not None:
    # highlight line
    folium.GeoJson(
        build_geojson(selected.to_frame().T),
        style_function=lambda f: {"color":"#000000", "weight":8, "opacity":1.0},
        name="Selected"
    ).add_to(m)
    # click marker
    folium.CircleMarker(location=[sel_lat, sel_lon], radius=5, color="#000000",
                        fill=True, fill_opacity=1).add_to(m)

# render & yeni tıklamayı yakala
out = st_folium(m, height=600, use_container_width=True, returned_objects=["last_object_clicked"])

# yeni tıklama geldiyse kaydet ve tekrar çiz
if out and out.get("last_object_clicked"):
    lat = float(out["last_object_clicked"]["lat"])
    lon = float(out["last_object_clicked"]["lng"])
    st.session_state.last_click = (lat, lon)
    st.rerun()

# ---------- form ----------
if selected is not None:
    st.subheader("Evaluate this segment")
    pred_label = str(selected.get("disturbance_label",""))
    pred_score = float(selected.get("disturbance",0.0))
    highway    = str(selected.get("highway",""))
    st.write(f"**Road:** {highway}  |  **Model:** {pred_label} (score={pred_score:.2f})")

    agree   = st.radio("Do you agree with the model’s prediction?", ["Yes","No"], horizontal=True)
    rating  = st.slider("Your perception (1 = very quiet, 5 = very noisy)", 1, 5, 3)
    comment = st.text_input("Optional comment (traffic, construction, etc.)")

    with st.expander("Optional: share your GPS coordinates"):
        user_lat = st.text_input("Latitude", "")
        user_lon = st.text_input("Longitude", "")

    if st.button("Submit"):
        payload = {
            "osmid": str(selected.get("osmid","")),
            "highway": highway,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "agree": agree,
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
                    st.success("✅ Thanks! Your response has been saved.")
                else:
                    st.error(f"Save failed: {js}")
            else:
                st.error(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            st.error(f"Connection error: {e}")
else:
    st.info("Haritaya bir kez tıklayıp segment seçin.")
