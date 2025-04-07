import torch
from dsets import Webcams
import os
from datetime import datetime
import numpy as np
import csv

class_values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
dset_dir = "C:\\Users\\sm380923\\Desktop\\Research\\good"
limits = dict()

start_date = datetime(2022, 1, 1)
end_date = datetime(2025, 12, 31)
visibility_range = (3.0, 7.0)  # Include visibilities >=3.0 and <=7.0
allowed_sites = None  

dset = Webcams.Webcams_cls_10(dset_dir, lambda x: x, limits)

info_list = []

for i, file in enumerate(dset.files):
    dir_name = os.path.basename(os.path.dirname(file))
    file_name = os.path.basename(file)

    try:
        timestamp = datetime.strptime(dir_name, '%Y-%m-%d-%H-%M-%S')
    except:
        timestamp = None

    name = file_name[:-4]
    site, ornt, _ = name.split('_')
    vis = class_values[torch.argmax(dset.labels[i]).item()]

    if start_date and timestamp and timestamp < start_date:
        continue
    if end_date and timestamp and timestamp > end_date:
        continue
    if visibility_range and not (visibility_range[0] <= vis <= visibility_range[1]):
        continue
    if allowed_sites and site not in allowed_sites:
        continue

    info_list.append((site, ornt, vis, timestamp))

#images per site
site_counts = {}
for site, _, _, _ in info_list:
    site_counts[site] = site_counts.get(site, 0) + 1

#visibility per site
site_vis_counts = {}
for site, _, vis, _ in info_list:
    if site not in site_vis_counts:
        site_vis_counts[site] = {v: 0 for v in class_values}
    site_vis_counts[site][vis] += 1

#histogram bin setup
bucket_edges = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
bucket_labels = [f"{bucket_edges[i]+1}-{bucket_edges[i+1]}" for i in range(len(bucket_edges)-1)]
bucket_labels[0] = f"{bucket_edges[0]}-{bucket_edges[1]}"

histogram_summary = {label: 0 for label in bucket_labels}
site_rows = []

for site in sorted(site_counts.keys()):
    count = site_counts[site]
    vis_breakdown = site_vis_counts.get(site, {v: 0 for v in class_values})

    # Determine histogram bin
    hist_label = "N/A"
    for i in range(len(bucket_edges) - 1):
        if bucket_edges[i] <= count <= bucket_edges[i + 1]:
            hist_label = bucket_labels[i]
            histogram_summary[hist_label] += 1
            break

    row = [site, count, hist_label] + [vis_breakdown[v] for v in class_values]
    site_rows.append(row)

with open('Site_Specific_Summary.csv', 'w') as f:
    writer = csv.writer(f)

    header = ['Site', 'Image Count', 'Histogram Bin'] + [f'Vis {v}' for v in class_values]
    writer.writerow(header)

    for row in site_rows:
        writer.writerow(row)

    writer.writerow([])

    writer.writerow(['Image Count Range', 'Number of Sites'])
    for label in bucket_labels:
        writer.writerow([label, histogram_summary[label]])

print("All-in-one CSV created: 'Site_Specific_Summary.csv'")