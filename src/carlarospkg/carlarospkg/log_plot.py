import os, csv
import matplotlib.pyplot as plt
import numpy as np

csv_path = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/latency_log.csv")
lat = []
with open(csv_path, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        if int(row["seq"]) == 1:
            continue
        lat.append(float(row["lat_service_ms"]))

lat_np = np.array(lat)

mean_lat = lat_np.mean()
p95_lat  = np.percentile(lat_np, 95)
p99_lat  = np.percentile(lat_np, 99)

print(f"Mean latency (ms): {mean_lat:.2f}")
print(f"P95 latency (ms): {p95_lat:.2f}")
print(f"P99 latency (ms): {p99_lat:.2f}")
print(f"Max latency (ms): {lat_np.max():.2f}")



plt.figure()
plt.hist(lat, bins=50, range=(min(lat) - 5, max(lat) + 5))
plt.axvline(p95_lat, linestyle="--", linewidth=2, label=f"P95 = {p95_lat:.1f} ms")
plt.axvline(p99_lat, linestyle=":", linewidth=2, label=f"P99 = {p99_lat:.1f} ms")
plt.ylabel("Count")
plt.xlabel("Service latency (ms)")
plt.title("Model Service Latency Histogram")
plt.legend()
plt.grid(False)
plt.tight_layout()

plot_path = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/latency_plot.png")
plt.savefig(plot_path, dpi=200)
print("Saved:", plot_path)
