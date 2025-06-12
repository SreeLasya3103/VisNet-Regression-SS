import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from difflib import get_close_matches


metrics_path = r"C:\\Users\\sm380923\\Desktop\\Research\\Visibility-Networks\\rewrite2\\All_Sites_Metrics.csv"
df_metrics = pd.read_csv(metrics_path)
stations_path = r"C:\\Users\\sm380923\\Desktop\\Research\\Visibility-Networks\\rewrite2\\stations.json"
with open(stations_path, 'r') as f:
    stations_data = json.load(f)

df = pd.read_csv(metrics_path)
df = df.groupby("site_id").agg({
    "accuracy": "mean",
    "macro_f1": "mean",
    "macro_precision": "mean",
    "macro_recall": "mean"
}).reset_index()

# Load FAA station info
with open(stations_path, "r") as f:
    stations = json.load(f)

station_df = pd.DataFrame(stations)

# Make sure you load the "stations" key if it exists, or just use the list
if isinstance(stations, dict) and "stations" in stations:
    station_df = pd.DataFrame(stations["stations"])
else:
    station_df = pd.DataFrame(stations)

# Check actual column names before renaming
print("Available columns in stations.json:")
print(station_df.columns)

# Normalize column names (adjust here if they differ)
if "site" in station_df.columns:
    station_df.rename(columns={"site": "site_name"}, inplace=True)

# Try to extract necessary columns
required_cols = ["site_name", "latitude", "longitude"]
station_df = station_df[[col for col in required_cols if col in station_df.columns]]


# Match site_id to station name using string similarity
matched_data = []

for site_id in df["site_id"]:
    matches = get_close_matches(site_id.lower(), station_df["site_name"].str.lower(), n=1, cutoff=0.6)
    if matches:
        matched_name = matches[0]
        station_row = station_df[station_df["site_name"].str.lower() == matched_name].iloc[0]
        row = df[df["site_id"] == site_id].iloc[0]
        matched_data.append({
            "site_id": site_id,
            "site_name": station_row["site_name"],
            "latitude": station_row["latitude"],
            "longitude": station_row["longitude"],
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "macro_precision": row["macro_precision"],
            "macro_recall": row["macro_recall"]
        })

# Create final DataFrame
final_df = pd.DataFrame(matched_data)

# Optional: Save matched results for debugging
final_df.to_csv("Matched_Site_Metrics.csv", index=False)

# Plot heatmap
plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    final_df["longitude"],
    final_df["latitude"],
    c=final_df["accuracy"],
    cmap="viridis",
    s=100,
    edgecolors="black"
)

plt.colorbar(scatter, label="Accuracy")
plt.title("FAA Site-Level Accuracy Heatmap")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.tight_layout()
plt.savefig("Heatmap_Final.png", dpi=300)
plt.show()