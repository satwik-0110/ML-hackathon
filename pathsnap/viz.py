"""Folium map builder shared by the EDA/maps scripts."""
import folium
import numpy as np
from branca.element import Element


def _arc_lines(arc_ids, net):
    a = np.asarray(arc_ids, dtype=np.int64)
    lon_a, lat_a = net.proj.to_lonlat(net.ax[a], net.ay[a])
    lon_b, lat_b = net.proj.to_lonlat(net.bx[a], net.by[a])
    return [[[la, lo], [lb, lob]] for la, lo, lb, lob in zip(lat_a, lon_a, lat_b, lon_b)]


def nearby_arcs(px, py, net, radius=80.0, max_arcs=15000):
    pi, arc, _, _ = net.arcs_near(px[::3], py[::3], radius)
    return np.unique(arc)[:max_arcs]


def legend_html(items):
    rows = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{t}</div>' for t, c in items)
    return Element(f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
                   f'padding:8px 10px;border:1px solid #999;border-radius:6px;font:13px sans-serif">{rows}</div>')


def trip_map(track, net, px, py, ref_route=None, clf_arcs=None, pred_route=None, title=""):
    lat, lon = track[:, 1], track[:, 0]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    items = []

    g = folium.FeatureGroup("Road network (near track)", show=True)
    for seg in _arc_lines(nearby_arcs(px, py, net), net):
        folium.PolyLine(seg, color="#9aa0a6", weight=1.5, opacity=0.7).add_to(g)
    g.add_to(m); items.append(("Road network", "#9aa0a6"))

    # GPS points are added before the route layers so the route lines are drawn on top of them
    g = folium.FeatureGroup("GPS observations", show=True)
    step = max(1, len(track) // 3000)
    for i in range(0, len(track), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2.5, color="#1f5fd6", fill=True, weight=0,
                            fill_opacity=0.6, tooltip=f"#{i} t={track[i, 2]:.0f}s").add_to(g)
    g.add_to(m); items.append(("GPS observations", "#1f5fd6"))

    if ref_route is not None:
        g = folium.FeatureGroup("Reference route", show=True)
        for seg in _arc_lines(ref_route, net):
            folium.PolyLine(seg, color="#2ca02c", weight=7, opacity=0.55).add_to(g)
        g.add_to(m); items.append(("Reference route (ground truth)", "#2ca02c"))

    if clf_arcs is not None:
        g = folium.FeatureGroup("Classifier-only arcs", show=False)
        for seg in _arc_lines(np.unique(clf_arcs[clf_arcs >= 0]), net):
            folium.PolyLine(seg, color="#ff7f0e", weight=3, opacity=0.9).add_to(g)
        g.add_to(m); items.append(("Classifier-only arcs", "#ff7f0e"))

    if pred_route is not None:
        g = folium.FeatureGroup("Predicted route (HMM)", show=True)
        for seg in _arc_lines(pred_route, net):
            folium.PolyLine(seg, color="#d62728", weight=3.5, opacity=0.95).add_to(g)
        g.add_to(m); items.append(("Predicted route (classifier + HMM)", "#d62728"))

    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        m.get_root().html.add_child(Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:4px 10px;'
            f'border-radius:6px;font:bold 14px sans-serif;border:1px solid #999">{title}</div>'))
    m.get_root().html.add_child(legend_html(items))
    return m
