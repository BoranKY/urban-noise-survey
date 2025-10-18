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

st.set_page_config(page_title="Urban Noise Survey", layout="wide")
st.title("🗺️ Urban Noise – Perception Survey")

# ========== CONFIG (Secrets) ==========
APPSCRIPT_URL = st.secrets["APPSCRIPT_URL"]      # "https://script.google.com/macros/s/XXX/exec"
APPSCRIPT_TOKEN = st.secrets["APPSCRIPT_TOKEN"]  # "YOUR_SHARED_TOKEN"

# ========== DATA LOADING ==========
@st.cache_data
def load_data(path):
    gdf = gpd.read_file(path).to_crs(4326)
    gdf["disturbance"] = pd.to_numeric(gdf["disturbance"], errors="coerce")
    if "disturbance_label" not in gdf.columns:
        bins = [0, 0.33, 0.66, 1]
        gdf["disturbance_label"] = pd.cut(
            gdf["disturbance"], bins=bins,
            labels=["Low", "Medium", "High"], include_lowest=True
        )
    return gdf

df = load_data("data/roads_wgs.geojson.gz")   # veya .geojson.gz

# ========== MAP ==========
center = [df.geometry.representative_point().y.mean(),
          df.geometry.representative_point().x.mean()]
m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")
cmap = cm.LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=0, vmax=1)

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
).add_to(m)
cmap.add_to(m)

st.caption("Click a road segment to evaluate its perceived noise.")
out = st_folium(m, height=600, use_container_width=True, returned_objects=["last_object_clicked"])

# ========== SELECTION ==========
selected = None
if out and out.get("last_object_clicked"):
    lat, lon = out["last_object_clicked"]["lat"], out["last_object_clicked"]["lng"]
    clicked_pt = gpd.GeoSeries([gpd.points_from_xy([lon],[lat])[0]], crs=4326).to_crs(df.crs)
    idx = df.distance(clicked_pt.iloc[0]).sort_values().index[0]
    selected = df.loc[idx]

# ========== SURVEY FORM ==========
if selected is not None:
    st.subheader("Evaluate this segment")
    st.write(f"**Road:** {selected['highway']}  |  **Model prediction:** {selected['disturbance_label']} (score={selected['disturbance']:.2f})")

    agree = st.radio("Do you agree with the prediction?", ["Yes", "No"], horizontal=True)
    rating = st.slider("Your perception (1 = very quiet, 5 = very noisy)", 1, 5, 3)
    comment = st.text_input("Optional comment (traffic, construction, etc.)")
    with st.expander("Optional: share your GPS coordinates"):
        user_lat = st.text_input("Latitude", "")
        user_lon = st.text_input("Longitude", "")

    if st.button("Submit"):
        payload = {
            "osmid": str(selected.get("osmid", "")),
            "highway": str(selected.get("highway", "")),
            "pred_label": str(selected["disturbance_label"]),
            "pred_score": float(selected["disturbance"]),
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

