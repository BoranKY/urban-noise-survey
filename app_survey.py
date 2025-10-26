import os
import json
import gzip
import uuid
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

# =========================
# Language (EN / FR)
# =========================
LANG = st.radio("Language / Langue", ["English", "Français"], horizontal=True)

T = {
    "English": {
        "title": "🗺️ Urban Noise – Perception Survey",
        "caption": "Click a road segment to evaluate its perceived noise.",
        "legend": "Disturbance Score (0–1)",
        "no_lines": "No line features to display after cleaning. Check your input data.",
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
        "thanks_sheet": "✅ Saved to Google Sheet and local backup.",
        "thanks_local": "✅ Saved to local backup. (Sheet save failed/slow)",
        "save_failed": "Save failed",
        "http_err": "HTTP {code}: {text}",
        "conn_err": "Connection error: {e}",
        "missing_secret": (
            "Missing secret: {k}\n\n"
            "Go to: Manage app → Settings → Secrets and add:\n"
            'APPSCRIPT_URL = "https://script.google.com/macros/s/.../exec"\n'
            'APPSCRIPT_TOKEN = "YOUR_SHARED_TOKEN"'
        ),
        "data_not_found": "Data file not found: {path}\n\nAvailable in ./data: {listing}\n",
        "backup_title": "Responses (local backup)",
        "backup_empty": "No local backup yet. Submit a response to see it here.",
        "download_csv": "⬇️ Download CSV (backup)",
    },
    "Français": {
        "title": "🗺️ Bruit urbain – Enquête de perception",
        "caption": "Cliquez sur un tronçon de route pour évaluer le bruit perçu.",
        "legend": "Indice de nuisance (0–1)",
        "no_lines": "Aucun tronçon à afficher après nettoyage. Vérifiez vos données.",
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
        "thanks_sheet": "✅ Enregistré dans Google Sheet et la sauvegarde locale.",
        "thanks_local": "✅ Enregistré dans la sauvegarde locale. (Échec/lenteur côté Sheet)",
        "save_failed": "Échec de l’enregistrement",
        "http_err": "HTTP {code} : {text}",
        "conn_err": "Erreur de connexion : {e}",
        "missing_secret": (
            "Secret manquant : {k}\n\n"
            "Allez dans : Manage app → Settings → Secrets et ajoutez :\n"
            'APPSCRIPT_URL = "https://script.google.com/macros/s/.../exec"\n'
            'APPSCRIPT_TOKEN = "YOUR_SHARED_TOKEN"'
        ),
        "data_not_found": "Fichier introuvable : {path}\n\nPrésents dans ./data : {listing}\n",
        "backup_title": "Réponses (sauvegarde locale)",
        "backup_empty": "Aucune sauvegarde locale pour l’instant. Envoyez une réponse pour l’afficher ici.",
        "download_csv": "⬇️ Télécharger le CSV (sauvegarde)",
    }
}[LANG]

st.title(T["title"])

# =========================
# Secrets guard
# =========================
def require_secret(key_name: str) -> str:
    val = st.secrets.get(key_name)
    if not val:
        st.error(T["missing_secret"].format(k=key_name))
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
        st.error(T["data_not_found"].format(path=path, listing=listing))
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
    st.error(T["no_lines"])
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
# Map (original colors)
# =========================
center = [
    df.geometry.representative_point().y.mean(),
    df.geometry.representative_point().x.mean()
]
m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

cmap = cm.LinearColormap(['green', 'yellow', 'orange', 'red'], vmin=0, vmax=1)
cmap.caption = T["legend"]

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
        aliases=(["Road", "Predicted Level", "Score"]
                 if LANG == "English" else ["Route", "Niveau prédit", "Score"])
    ),
    name="Roads"
).add_to(m)
cmap.add_to(m)

st.caption(T["caption"])
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
# Local backup helpers (repo root: responses_backup.csv)
# =========================
BACKUP_CSV = "responses_backup.csv"
COLUMNS = [
    "timestamp","uuid","osmid","highway","pred_label","pred_score",
    "agree","rating_1to5","comment","click_lat","click_lon","user_lat","user_lon"
]

def append_local_backup(row_dict: dict):
    """Append to responses_backup.csv (create with header if missing)."""
    file_exists = os.path.exists(BACKUP_CSV)
    df_row = pd.DataFrame([row_dict], columns=COLUMNS)
    if not file_exists:
        df_row.to_csv(BACKUP_CSV, index=False)
    else:
        df_row.to_csv(BACKUP_CSV, mode="a", header=False, index=False)

def load_backup_df() -> pd.DataFrame:
    if os.path.exists(BACKUP_CSV):
        try:
            return pd.read_csv(BACKUP_CSV)
        except Exception:
            return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)

# =========================
# Survey form
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
        # Normalize FR Yes/No to EN
        agree_norm = {"Yes": "Yes", "No": "No", "Oui": "Yes", "Non": "No"}.get(agree, str(agree))

        req_uuid = str(uuid.uuid4())
        ts = datetime.utcnow().isoformat()

        payload = {
            "uuid": req_uuid,
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

        # 1) Always write local backup
        local_row = {"timestamp": ts, **payload}
        append_local_backup(local_row)

        # 2) Try sending to Apps Script (timeout=30s)
        try:
            r = requests.post(
                APPSCRIPT_URL,
                params={"token": APPSCRIPT_TOKEN},
                json=payload,
                timeout=30
            )
            if r.ok:
                try:
                    resp = r.json()
                except Exception:
                    resp = {"status": "?", "raw": r.text[:200]}
                if resp.get("status") == "ok":
                    st.success(T["thanks_sheet"])
                else:
                    st.warning(f"{T['thanks_local']}  ({T['save_failed']}: {resp})")
            else:
                st.warning(T["thanks_local"] + "  " + T["http_err"].format(code=r.status_code, text=r.text[:200]))
        except Exception as e:
            st.warning(T["thanks_local"] + "  " + T["conn_err"].format(e=e))
else:
    st.info(T["click_info"])

# =========================
# Backup table + download
# =========================
st.divider()
st.subheader(T["backup_title"])
backup_df = load_backup_df()
st.dataframe(backup_df, use_container_width=True, height=300)

if not backup_df.empty:
    csv_bytes = backup_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=T["download_csv"],
        data=csv_bytes,
        file_name="responses_backup.csv",
        mime="text/csv"
    )
else:
    st.caption(T["backup_empty"])
