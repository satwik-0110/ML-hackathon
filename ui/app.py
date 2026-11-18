"""PathSnap evaluator UI (Laptop 2).

Lists the evaluation trips served by the API, sends the selected trip id to
POST /predict and displays the returned predicted route and metrics.
The UI never receives or displays the reference route.

Run:  streamlit run ui/app.py   (set PATHSNAP_API_URL=http://<laptop-1-ip>:8000)
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", page_icon="🛣️", layout="wide")
st.title("PathSnap — map-matching & route reconstruction")

api = st.sidebar.text_input("API URL (Laptop 1)", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000"))
try:
    h = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"API online · {h['model']} · {h['n_trips']} trips")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"API not reachable: {e}")
    st.stop()


@st.cache_data(ttl=60)
def get_trips(url):
    return requests.get(f"{url}/trips", timeout=30).json()["trips"]


trips = get_trips(api)
labels = {f"{t['trip_id']}  ·  {t['n_points']} pts  ·  {t['duration_s'] / 60:.0f} min": t["trip_id"] for t in trips}
choice = st.sidebar.radio("Evaluation trips", list(labels))
show_clf = st.sidebar.checkbox("Show classifier-only arcs (before HMM)", value=False)

if st.sidebar.button("Run PathSnap on this trip", type="primary", use_container_width=True):
    with st.spinner("API is processing the trip (candidates → classifier → HMM/Viterbi)…"):
        r = requests.post(f"{api}/predict", json={"trip_id": labels[choice]}, timeout=600)
    if r.ok:
        st.session_state["result"] = r.json()
    else:
        st.error(f"API error {r.status_code}: {r.text}")

res = st.session_state.get("result")
if not res:
    st.info("Select a trip on the left and press **Run PathSnap**.")
    st.stop()

m = res["metrics"]
st.subheader(f"Trip {res['trip_id']} · {res['n_points']} GPS points · model: {res['model']}")
c = st.columns(5)
if "top1_hmm" in m:
    c[0].metric("Point accuracy (HMM)", f"{m['top1_hmm']:.2%}",
                f"{(m['top1_hmm'] - m['top1_classifier']) * 100:+.2f} pp vs classifier")
    c[1].metric("Classifier-only accuracy", f"{m['top1_classifier']:.2%}")
    c[2].metric("Route F1 (length)", f"{m['route_f1']:.3f}")
    c[3].metric("Route precision / recall", f"{m['route_precision']:.3f} / {m['route_recall']:.3f}")
c[4].metric("Route continuity", f"{m['continuity_hmm']:.3f}", f"classifier-only {m['continuity_classifier']:.3f}",
            delta_color="off")

gps = res["gps"]
fmap = folium.Map(tiles="OpenStreetMap", control_scale=True)
lats, lons = [p[0] for p in gps], [p[1] for p in gps]
fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
g = folium.FeatureGroup("GPS observations")
for p in gps:
    folium.CircleMarker(p, radius=2, color="#1f5fd6", weight=0, fill=True, fill_opacity=0.8).add_to(g)
g.add_to(fmap)
if show_clf:
    g = folium.FeatureGroup("Classifier-only arcs")
    for seg in res["classifier_only_segments"]:
        folium.PolyLine(seg, color="#ff7f0e", weight=3).add_to(g)
    g.add_to(fmap)
if res["predicted_route"]:
    folium.PolyLine(res["predicted_route"], color="#d62728", weight=4, opacity=0.9,
                    tooltip="Predicted route").add_to(folium.FeatureGroup("Predicted route (HMM)").add_to(fmap))
folium.Marker(gps[0], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
folium.Marker(gps[-1], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
folium.LayerControl(collapsed=False).add_to(fmap)
st.caption("🔵 GPS observations   🔴 predicted route (classifier + HMM/Viterbi)   🟠 classifier-only arcs (optional)")
st_folium(fmap, height=620, use_container_width=True, returned_objects=[])

c1, c2 = st.columns(2)
c1.write("**All metrics (computed on the API server)**"); c1.json(m)
c2.write("**Diagnostics & timing**"); c2.json({**res["diagnostics"], "timing_s": res["timing_s"],
                                              "hmm_params": res["hmm_params"]})
