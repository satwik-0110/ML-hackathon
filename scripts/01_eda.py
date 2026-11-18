"""Exploratory data analysis of the raw dataset (train + student_test partitions).

Outputs:
  results/eda_trip_stats.csv     per-trip inventory and quality statistics
  results/eda_summary.txt        partition-level summary
  results/plots/eda_*.png        distributions, trip locations, example trajectories
  results/maps/eda_<trip>.html   Folium map: GPS points, road network, reference route
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from pathsnap.features import prepare_trip  # noqa: E402
from pathsnap.geo import point_segment  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.viz import trip_map  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS, MAPS = RESULTS_DIR / "plots", RESULTS_DIR / "maps"


def trip_stats(part, tid):
    folder = DATA_DIR / part / tid
    raw = np.loadtxt(folder / f"{tid}.track", ndmin=2)
    trip = load_trip(folder)
    tr = trip.track
    proj, px, py, net = prepare_trip(tr, trip.nodes, trip.arcs)
    dt = np.diff(tr[:, 2])
    step = np.hypot(np.diff(px), np.diff(py))
    V = len(trip.nodes)
    A = coo_matrix((np.ones(len(trip.arcs)), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(V, V))
    n_weak, _ = connected_components(A, directed=True, connection="weak")
    n_strong, lab_s = connected_components(A, directed=True, connection="strong")
    largest_scc = np.bincount(lab_s).max() / V
    used = np.zeros(V, bool); used[trip.arcs.ravel()] = True
    r = trip.route
    ra, rb = trip.arcs[r, 0], trip.arcs[r, 1]
    dmin = np.concatenate([
        point_segment(px[s:s + 500, None], py[s:s + 500, None], net.nx[ra][None], net.ny[ra][None],
                      net.nx[rb][None], net.ny[rb][None])[0].min(1) for s in range(0, len(px), 500)])
    return dict(
        partition=part, trip=tid, n_obs_raw=len(raw), n_obs_clean=len(tr),
        nan_rows=int((~np.isfinite(raw)).any(1).sum()), non_increasing_ts=int((np.diff(raw[:, 2]) <= 0).sum()),
        duration_s=tr[-1, 2] - tr[0, 2], frac_dt_1s=float(np.mean(dt == 1)), max_gap_s=float(dt.max()),
        n_gaps_over_10s=int((dt > 10).sum()), median_speed_ms=float(np.median(step[dt > 0] / dt[dt > 0])),
        max_step_m=float(step.max()), lon_center=float(np.median(tr[:, 0])), lat_center=float(np.median(tr[:, 1])),
        extent_km=float(np.hypot(np.ptp(px), np.ptp(py)) / 1000),
        n_nodes=V, n_arcs=len(trip.arcs), isolated_nodes=int((~used).sum()),
        weak_components=int(n_weak), strong_components=int(n_strong), largest_scc_frac=float(largest_scc),
        two_way_arc_frac=float(net.has_reverse.mean()), median_arc_len_m=float(np.median(net.length)),
        route_arcs=len(r), route_len_km=float(net.length[r].sum() / 1000),
        route_continuity=float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if len(r) > 1 else 1.0,
        gps_to_route_median_m=float(np.median(dmin)), gps_to_route_p95_m=float(np.percentile(dmin, 95)),
        frac_gps_over_50m=float(np.mean(dmin > 50)),
    ), dmin, dt


def main():
    PLOTS.mkdir(parents=True, exist_ok=True); MAPS.mkdir(parents=True, exist_ok=True)
    rows, dists, dts = [], [], []
    for part in ["train", "student_test"]:
        for tid in list_trips(DATA_DIR, part):
            s, dmin, dt = trip_stats(part, tid)
            rows.append(s); dists.append(dmin); dts.append(dt)
            print(part, tid, s["n_obs_clean"], f"{s['gps_to_route_median_m']:.1f}m", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "eda_trip_stats.csv", index=False)

    with open(RESULTS_DIR / "eda_summary.txt", "w") as f:
        f.write(df.groupby("partition").agg(trips=("trip", "count"), obs=("n_obs_clean", "sum"),
                                            median_obs=("n_obs_clean", "median"), nan_rows=("nan_rows", "sum"),
                                            bad_ts=("non_increasing_ts", "sum")).to_string() + "\n\n")
        f.write(df.describe().T.round(3).to_string() + "\n\n")
        f.write("Trips with GPS far from the reference route (median > 20 m):\n")
        f.write(df[df.gps_to_route_median_m > 20][["partition", "trip", "gps_to_route_median_m"]].to_string() + "\n")
        f.write("\nTrips with gaps > 300 s:\n" + df[df.max_gap_s > 300][["partition", "trip", "max_gap_s"]].to_string())
        f.write("\n\nTrips with reference-route discontinuities:\n" +
                df[df.route_continuity < 1][["partition", "trip", "route_continuity"]].to_string())

    tr = df[df.partition == "train"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(tr.n_obs_clean, bins=30); ax[0, 0].set_title("GPS observations per trip (train)")
    alldt = np.concatenate(dts)
    ax[0, 1].hist(np.clip(alldt, 0, 20), bins=40, log=True); ax[0, 1].set_title("Sampling interval (s, clipped at 20)")
    alld = np.concatenate(dists)
    ax[0, 2].hist(np.clip(alld, 0, 60), bins=60); ax[0, 2].set_title("Distance GPS -> reference route (m, clipped 60)")
    ax[1, 0].hist(np.log10(tr.n_arcs), bins=30); ax[1, 0].set_title("log10 #arcs per trip network")
    ax[1, 1].hist(tr.median_speed_ms * 3.6, bins=30); ax[1, 1].set_title("Median speed per trip (km/h)")
    ax[1, 2].scatter(df.lon_center, df.lat_center, c=(df.partition == "train"), cmap="coolwarm", s=18)
    ax[1, 2].set_title("Trip locations (lon/lat): red=train, blue=student test")
    plt.tight_layout(); plt.savefig(PLOTS / "eda_overview.png", dpi=120); plt.close()

    # example trajectories in projected metres
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, tid in zip(ax, ["00000001", "00000014", "00000033", "00000090"]):
        t = load_trip(DATA_DIR / "train" / tid)
        _, px, py, net = prepare_trip(t.track, t.nodes, t.arcs)
        r = t.route
        for k in r:
            a.plot([net.ax[k], net.bx[k]], [net.ay[k], net.by[k]], color="green", lw=2)
        a.plot(px, py, ".", ms=1.5, color="tab:blue")
        a.set_title(f"train {tid}"); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    plt.suptitle("GPS points (blue) vs reference route (green); 00000033 shows a systematic offset")
    plt.tight_layout(); plt.savefig(PLOTS / "eda_trajectories.png", dpi=120); plt.close()

    for part, tid in [("train", "00000001"), ("train", "00000033")]:
        t = load_trip(DATA_DIR / part / tid)
        _, px, py, net = prepare_trip(t.track, t.nodes, t.arcs)
        trip_map(t.track, net, px, py, ref_route=t.route, title=f"EDA - {part} {tid}").save(MAPS / f"eda_{tid}.html")
    print(open(RESULTS_DIR / "eda_summary.txt").read()[:3000])


if __name__ == "__main__":
    main()
