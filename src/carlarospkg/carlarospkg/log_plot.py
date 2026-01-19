# import os, csv
# import matplotlib.pyplot as plt
# import numpy as np

# csv_path = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/latency_log.csv")
# lat = []
# with open(csv_path, "r") as f:
#     r = csv.DictReader(f)
#     for row in r:
#         if int(row["seq"]) == 1:
#             continue
#         lat.append(float(row["lat_service_ms"]))

# lat_np = np.array(lat)

# mean_lat = lat_np.mean()
# p95_lat  = np.percentile(lat_np, 95)
# p99_lat  = np.percentile(lat_np, 99)

# print(f"Mean latency (ms): {mean_lat:.2f}")
# print(f"P95 latency (ms): {p95_lat:.2f}")
# print(f"P99 latency (ms): {p99_lat:.2f}")
# print(f"Max latency (ms): {lat_np.max():.2f}")



# plt.figure()
# plt.hist(lat, bins=50, range=(min(lat) - 5, max(lat) + 5))
# plt.axvline(p95_lat, linestyle="--", linewidth=2, label=f"P95 = {p95_lat:.1f} ms")
# plt.axvline(p99_lat, linestyle=":", linewidth=2, label=f"P99 = {p99_lat:.1f} ms")
# plt.ylabel("Count")
# plt.xlabel("Service latency (ms)")
# plt.title("Model Service Latency Histogram")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()

# plot_path = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/latency_plot.png")
# plt.savefig(plot_path, dpi=200)
# print("Saved:", plot_path)

import os, csv
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# UPDATE THESE FOUR PATHS
# ----------------------------

# ROS bridge: Jetson prediction-time CSV (ms)
ROS_PRED_CSV = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/RCB_prediction_log.csv")
# Single system: prediction-time CSV (seconds)
SINGLE_CSV   = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/SS_Total_and_Pred_log.csv")

# ROS bridge: total elapsed-time CSV (seconds or ms - set flag below)
ROS_ELAPSED_CSV = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/RCB_Total_Time_log.csv")

# Output directory for plots
OUT_DIR = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# HELPERS
# ----------------------------
def load_col(csv_path, col, skip_first=True):
    vals = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            # optional warmup skip if seq/iter exists
            if skip_first:
                if "seq" in row and row["seq"] and int(row["seq"]) == 1:
                    continue
                if "iter" in row and row["iter"] and int(row["iter"]) == 1:
                    continue
            if row.get(col, "") == "":
                continue
            vals.append(float(row[col]))
    return np.array(vals, dtype=float)

def print_stats(name, arr_ms):
    print(f"\n{name}")
    print(f"  N   : {len(arr_ms)}")
    print(f"  mean: {arr_ms.mean():.2f} ms")
    print(f"  p95 : {np.percentile(arr_ms, 95):.2f} ms")
    print(f"  p99 : {np.percentile(arr_ms, 99):.2f} ms")
    print(f"  max : {arr_ms.max():.2f} ms")

def overlay_hist(a_ms, b_ms,label_a, label_b, title, xlabel, out_png,bins=60):

    # stats
    mean_a = a_ms.mean()
    mean_b = b_ms.mean()

    p95_a = np.percentile(a_ms, 95)
    p95_b = np.percentile(b_ms, 95)

    p99_a = np.percentile(a_ms, 99)
    p99_b = np.percentile(b_ms, 99)

    xmax = float(max(a_ms.max(), b_ms.max()))

    plt.figure()

    # histograms
    plt.hist(
        a_ms, bins=bins, range=(0, xmax),
        histtype="step", linewidth=2,
        label=(
            f"{label_a}\n"
            f"mean={mean_a:.1f} ms | P95={p95_a:.1f} | P99={p99_a:.1f}"
        )
    )

    plt.hist(
        b_ms, bins=bins, range=(0, xmax),
        histtype="step", linewidth=2,
        label=(
            f"{label_b}\n"
            f"mean={mean_b:.1f} ms | P95={p95_b:.1f} | P99={p99_b:.1f}"
        )
    )

    # vertical lines — ROS
    plt.axvline(mean_a, linestyle="-",  linewidth=1.5)
    plt.axvline(p95_a, linestyle="--", linewidth=1.5)
    plt.axvline(p99_a, linestyle=":",  linewidth=1.5)

    # vertical lines — Single system
    plt.axvline(mean_b, linestyle="-",  linewidth=1.5)
    plt.axvline(p95_b, linestyle="--", linewidth=1.5)
    plt.axvline(p99_b, linestyle=":",  linewidth=1.5)

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.xlim(0, xmax)
    plt.grid(False)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print("Saved:", out_png)


# ----------------------------
# 1) PREDICTION TIME HISTOGRAM
# ----------------------------

# ROS bridge prediction time: already in ms
ros_pred_ms = load_col(ROS_PRED_CSV, "lat_service_ms")

# Single-system prediction time: in seconds -> convert to ms
single_pred_s = load_col(SINGLE_CSV, "pred_time")
single_pred_ms = single_pred_s * 1000.0

print_stats("ROS bridge prediction time (Jetson lat_service_ms)", ros_pred_ms)
print_stats("Single-system prediction time (pred_time)", single_pred_ms)

overlay_hist(
    ros_pred_ms, single_pred_ms,
    "ROS bridge (Jetson service predict time)",
    "Single system (local predict time)",
    title="Prediction Time Histogram: ROS Bridge vs Single System",
    xlabel="Prediction time (ms)",
    out_png=os.path.join(OUT_DIR, "hist_prediction_overlay.png"),
    bins=60
)



# ----------------------------
# 2) TOTAL ELAPSED TIME HISTOGRAM
# ----------------------------

# ROS elapsed time: assume seconds -> convert to ms
# If your ROS elapsed CSV is already ms, set ROS_ELAPSED_IS_SECONDS = False
ROS_ELAPSED_IS_SECONDS = False

# Column names expected:
# - ROS elapsed CSV: elapsed_time (seconds) OR elapsedtime_s, etc.
#   Update the column name below if needed.
ROS_ELAPSED_COL = "lat_service_ms"   # change if your header is different

ros_elapsed = load_col(ROS_ELAPSED_CSV, ROS_ELAPSED_COL)
ros_elapsed_ms = ros_elapsed * 1000.0 if ROS_ELAPSED_IS_SECONDS else ros_elapsed

# Single-system elapsed_time: seconds -> ms
single_elapsed_s = load_col(SINGLE_CSV, "elapsed_time")
single_elapsed_ms = single_elapsed_s * 1000.0

print_stats("ROS bridge total elapsed time", ros_elapsed_ms)
print_stats("Single-system total elapsed time", single_elapsed_ms)

overlay_hist(
    ros_elapsed_ms, single_elapsed_ms,
    "ROS bridge (total step time)",
    "Single system (total step time)",
    title="Total Elapsed Time Histogram: ROS Bridge vs Single System",
    xlabel="Elapsed time per step (ms)",
    out_png=os.path.join(OUT_DIR, "hist_elapsed_overlay.png"),
    bins=60
)


