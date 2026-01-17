import os, csv
import matplotlib.pyplot as plt

csv_path = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/latency_log.csv")
lat = []
with open(csv_path, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        if int(row["seq"]) == 1:
            continue
        lat.append(float(row["lat_service_ms"]))
print(max(lat))

plt.figure()
plt.hist(lat, bins=50, range=(min(lat) - 5, max(lat) + 5))
plt.ylabel("Count")
plt.xlabel("Service latency (ms)")
plt.title("Model Service Latency Histogram")
plt.grid(False)
plt.tight_layout()

plot_path = os.path.expanduser("/home/krg6/Capstone/PilotNet_Train/pilotnetros/latency_logs/latency_plot.png")
plt.savefig(plot_path, dpi=200)
print("Saved:", plot_path)
