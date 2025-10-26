# =========================
# Map (configurable heat view)
# =========================
from folium import plugins

# ---- Sidebar controls
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

# ---- Filtered dataframe
vis = df.copy()
if highway_filter:
    vis = vis[vis["highway"].astype(str).isin(highway_filter)]
vis = vis[vis["disturbance"] >= float(min_score)]
if vis.empty:
    st.warning("No features match current filters.")
    st.stop()

# ---- Map center
center = [
    vis.geometry.representative_point().y.mean(),
    vis.geometry.representative_point().x.mean()
]

m = folium.Map(location=center, zoom_start=map_zoom, tiles="cartodbpositron")

# ---- Colormap
if cmap_name == "blue→purple→red":
    ramp = ['#2c7fb8', '#7b3294', '#d7191c']
else:
    ramp = ['green', 'yellow', 'red']

# optionally classify values
vals = vis["disturbance"].astype(float).clip(0, 1)
if scheme.startswith("quantile"):
    q = np.quantile(vals, [0, .2, .4, .6, .8, 1])
    def scaler(v):  # map to [0,1] by quantile rank
        return (np.searchsorted(q, v, side="right")-1)/5
elif scheme.startswith("equal"):
    bins = np.linspace(vals.min(), vals.max(), 6)
    def scaler(v):
        return (np.searchsorted(bins, v, side="right")-1)/5
else:  # continuous
    vmin, vmax = float(vals.min()), float(vals.max())
    span = max(vmax-vmin, 1e-9)
    def scaler(v): return (float(v)-vmin)/span

cmap = cm.LinearColormap(ramp, vmin=0, vmax=1)
cmap.caption = "Disturbance (scaled)"

# ---- LAYER 1: colored lines
if show_line:
    gj = folium.GeoJson(
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
    )
    gj.add_to(m)
    cmap.add_to(m)

# ---- LAYER 2: point heatmap (weighted)
if show_heat:
    # one representative point per segment (lightweight). 
    # İstersen daha yoğun yapmak için her LineString için birkaç noktayı interpolate edebilirsin.
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
            radius=18,     # daha pürüzsüz görünüm için ayarla
            blur=15,
            max_zoom=18,
            min_opacity=0.3
        ).add_to(m)

# ---- Useful controls
plugins.Fullscreen(position="topleft").add_to(m)
plugins.MeasureControl(primary_length_unit='meters').add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st.caption("Tip: Use the sidebar to change classification, filters and layers.")
out = st_folium(m, height=650, use_container_width=True, returned_objects=["last_object_clicked"])
