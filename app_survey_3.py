import os
import json
import gzip
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from shapely.geometry import Point, mapping
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
import requests  # Google Form'a POST için

# (Optional) get user geolocation; if missing, we fall back to blanks
try:
    from streamlit_js_eval import get_geolocation
    HAS_JS_GEO = True
except Exception:
    HAS_JS_GEO = False

# =========================
#  Google Form settings
# =========================

FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSfsz-S7yQrXomUStbgKJHZyzIp35xLWip3zsOH2aPNo5A9RZg/formResponse"
)

ENTRY_MAP = {
    "osmid":             "entry.1390619366",
    "highway":           "entry.2001035176",
    "pred_label":        "entry.1555792867",
    "pred_score":        "entry.511753294",
    "rating_1to5":       "entry.221820768",
    "click_lat":         "entry.746558089",
    "click_lon":         "entry.1412698142",
    "name":              "entry.1014633606",
    "age":               "entry.1860211938",
    "gender":            "entry.736143174",
    "years_fribourg":    "entry.1877525660",
    "role":              "entry.2111502710",
    "noise_sensitivity": "entry.956148229",
}


def send_to_google_form(response: dict, timeout: int = 20):
    """
    Streamlit'te topladığımız response dict'ini Google Form'a POST eder.
    200 veya 302 dönerse 'başarılı' kabul ediyoruz.
    """
    data = {
        ENTRY_MAP["osmid"]:             str(response.get("osmid", "")),
        ENTRY_MAP["highway"]:           str(response.get("highway", "")),
        ENTRY_MAP["pred_label"]:        str(response.get("pred_label", "")),
        ENTRY_MAP["pred_score"]:        str(response.get("pred_score", "")),
        ENTRY_MAP["rating_1to5"]:       str(response.get("rating_1to5", "")),
        ENTRY_MAP["click_lat"]:         str(response.get("click_lat", "")),
        ENTRY_MAP["click_lon"]:         str(response.get("click_lon", "")),
        ENTRY_MAP["name"]:              str(response.get("name", "")),
        ENTRY_MAP["age"]:               str(response.get("age", "")),
        ENTRY_MAP["gender"]:            str(response.get("gender", "")),
        ENTRY_MAP["years_fribourg"]:    str(response.get("years_fribourg", "")),
        ENTRY_MAP["role"]:              str(response.get("role", "")),
        ENTRY_MAP["noise_sensitivity"]: str(response.get("noise_sensitivity_1to5", "")),
    }

    r = requests.post(FORM_URL, data=data, timeout=timeout)
    return r.status_code in (200, 302), r.status_code, r.text[:200]


# =========================
#  Page & Header
# =========================
st.set_page_config(page_title="How Noisy Is This Street?", layout="wide")
st.title("How Noisy Is This Street?")

st.markdown("""
**Help us map where streets feel quiet or noisy.** Your feedback helps us understand
how well our noise predictions match what people actually feel on the ground.

**Colors on the map (model prediction):**
- 🟢 **Green** = Low (quieter)
- 🟡 **Yellow / Orange** = Medium
- 🔴 **Red** = High (noisier)

**How it works (takes ~2–3 minutes):**
1. **Zoom** to an area you know well.
2. **Click** a street line on the map.
3. In the sidebar:
   - Fill in a few short questions **about yourself**.
   - Rate **how much you agree** with the noise prediction for that street.
   - (Optional) Add a short comment.
4. Press **“Submit response”**.

**Important:**  
- Please only rate streets that you know reasonably well.  
- There is no right or wrong answer – we are interested in **your perception**.
""")

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
            f"Tip: put your file under ./data and update DF_PATH."
        )
        st.stop()

    # LFS pointer check
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            st.error(
                "⚠️ The file seems to be a Git LFS pointer, not the actual GeoJSON.\n\n"
                "Fix: commit a simplified GeoJSON or gzipped GeoJSON under 25MB."
            )
            st.stop()
    except Exception:
        pass

    # Read file WITHOUT using geopandas.read_file/fiona
    try:
        if path.endswith(".geojson.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)

        if isinstance(data, dict) and "features" in data:
            gdf = gpd.GeoDataFrame.from_features(data["features"])
        else:
            st.error("❌ File is not a valid GeoJSON FeatureCollection.")
            st.stop()

        if gdf.crs is None:
            gdf.set_crs(4326, inplace=True)
        else:
            gdf = gdf.to_crs(4326)

    except Exception as e:
        st.error(f"❌ Failed to read geo data as GeoJSON: {path}\n\n{e}")
        st.stop()

    # Geometry cleaning
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


@st.cache_resource
def get_sindex(_gdf: gpd.GeoDataFrame):
    """Create a spatial index for fast nearest-segment queries."""
    try:
        return _gdf.sindex
    except Exception:
        return None


# =========================
#  Load data (cached)
# =========================
DF_PATH = "data/roads_wgs_fribourg_bbox.geojson"  # küçük bbox dosyan

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
m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

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

st.caption(
    "Click a street line on the map, then use the sidebar to answer the questions "
    "and submit your feedback."
)
out = st_folium(
    m,
    height=600,
    use_container_width=True,
    returned_objects=["last_object_clicked"],
)

# =========================
# Selection (nearest segment) + optional geolocation
# =========================
selected = None
lat = lon = None

if out and out.get("last_object_clicked"):
    lat = float(out["last_object_clicked"]["lat"])
    lon = float(out["last_object_clicked"]["lng"])
    click_geom = Point(lon, lat)  # df already in 4326

    if sindex is not None:
        try:
            nearest_idx = sindex.nearest(click_geom, return_all=False)
            tree_idx = None
            if hasattr(nearest_idx, "shape") and nearest_idx.shape[0] == 2:
                tree_idx = int(nearest_idx[1][0])
            else:
                idx_list = list(nearest_idx)
                tree_idx = int(idx_list[0])
            selected = df.iloc[tree_idx]
        except Exception:
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
# Sidebar: Demographics + Noise sensitivity + Street rating
# =========================
st.sidebar.header("About you")

name = st.sidebar.text_input("Name and surname *", key="name")
age = st.sidebar.number_input(
    "How old are you? *",
    min_value=18,
    max_value=100,
    value=25,
    step=1,
    key="age",
)

gender = st.sidebar.selectbox(
    "What is your gender?",
    options=[
        "Prefer not to say",
        "Female",
        "Male",
        "Non-binary / Other",
    ],
    index=0,
    key="gender",
)

years_fribourg = st.sidebar.selectbox(
    "How long have you been living in Fribourg? *",
    options=[
        "I don’t live in Fribourg",
        "Less than 1 year",
        "1–3 years",
        "3–5 years",
        "More than 5 years",
    ],
    index=0,
    key="years_fribourg",
)

role = st.sidebar.selectbox(
    "Which of the following best describes you? *",
    options=[
        "Local resident",
        "Student",
        "Commuter (I work here but live elsewhere)",
        "Tourist / visitor",
        "Other",
    ],
    index=1,
    key="role",
)

noise_sensitivity = st.sidebar.slider(
    "In general, how sensitive are you to everyday city noise? *",
    min_value=1,
    max_value=5,
    value=3,
    help=(
        "People differ a lot in how quickly they feel annoyed or stressed by "
        "noise (traffic, crowds, construction, etc.). "
        "Please rate your own sensitivity in daily life, compared to an "
        "average person of your age.\n\n"
        "1 = very low (I rarely get bothered by noise)\n"
        "5 = very high (noise bothers me very easily)."
    ),
    key="noise_sensitivity_1to5",
)

st.sidebar.markdown("---")
st.sidebar.header("Rate this street")

if selected is None:
    st.sidebar.info("Click a street on the map to start rating.")
else:
    pred_label = str(selected.get('disturbance_label', ''))
    pred_score = float(selected.get('disturbance', 0.0))
    highway = str(selected.get('highway', ''))

    st.sidebar.write(
        f"**Selected street:** `{highway}`  \n"
        f"**Model prediction:** {pred_label} "
        f"({pred_score:.2f} on [0, 1])"
    )

    rating = st.sidebar.slider(
        "How much do you agree with this street’s noise prediction? "
        "(1 = don’t agree at all, 5 = completely agree) *",
        1,
        5,
        3,
        key="rating_1to5",
    )

    if rating <= 2:
        st.sidebar.warning(
            "You seem to disagree. Please tell us why, and which color you think "
            "this street should be (green / yellow / red)."
        )

    comment = st.sidebar.text_input(
        "Optional comment (e.g., road works, rush hour, local context)",
        key="comment",
    )

    submit = st.sidebar.button("Submit response", use_container_width=True)

    if submit:
        errors = []
        if not name.strip():
            errors.append("Please enter your name and surname.")
        if selected is None:
            errors.append("Please click a street on the map.")

        if errors:
            st.sidebar.error("Please fix the following:\n- " + "\n- ".join(errors))
        else:
            response = {
                "timestamp": datetime.utcnow().isoformat(),
                "name": name.strip(),
                "age": int(age),
                "gender": gender,
                "years_fribourg": years_fribourg,
                "role": role,
                "noise_sensitivity_1to5": int(noise_sensitivity),
                "osmid": str(selected.get("osmid", "")),
                "highway": highway,
                "pred_label": pred_label,
                "pred_score": pred_score,
                "rating_1to5": int(rating),
                "comment": comment,
                "click_lat": float(lat) if lat is not None else "",
                "click_lon": float(lon) if lon is not None else "",
                "user_lat": user_lat,
                "user_lon": user_lon,
            }

            ok, code, preview = send_to_google_form(response, timeout=20)

            if ok:
                st.sidebar.success("✅ Thank you! Your response has been submitted.")
            else:
                st.sidebar.warning(
                    f"Your response could not be sent (HTTP {code}). "
                    "You can try again later."
                )

            st.subheader("Current response (for debugging)")
            st.json(response)
