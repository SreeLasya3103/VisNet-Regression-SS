import torch
from dsets import Webcams
import os
from datetime import datetime
import csv

class_values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
dset_dir = "D:\\Research\\NewGoodOnlyWebcams"
limits = dict()

start_date = datetime(2022, 1, 1)
end_date = datetime(2025, 12, 31)
visibility_range = (3.0, 7.0)
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

#Count images per site
site_counts = {}
for site, _, _, _ in info_list:
    site_counts[site] = site_counts.get(site, 0) + 1

#histogram bins
bucket_edges = [0, 12, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
bucket_labels = [f"{bucket_edges[i]+1}-{bucket_edges[i+1]}" for i in range(len(bucket_edges)-1)]
bucket_labels[0] = f"{bucket_edges[0]}-{bucket_edges[1]}"

histogram_summary = {label: 0 for label in bucket_labels}

#Count number of sites in each range
for count in site_counts.values():
    for i in range(len(bucket_edges) - 1):
        if bucket_edges[i] <= count <= bucket_edges[i + 1]:
            histogram_summary[bucket_labels[i]] += 1
            break

#sites and their number of images
with open('Site_Image_Counts.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Site', 'Number of Images'])
    for site, count in sorted(site_counts.items()):
        writer.writerow([site, count])

print("CSV created: 'Site_Image_Counts.csv'")

#Image count range and number of sites
#with open('Site_Specific_Histogram.csv', 'w') as f:
    #writer = csv.writer(f)
    #writer.writerow(['Image Count Range', 'Number of Sites'])
    #for label in bucket_labels:
        #writer.writerow([label, histogram_summary[label]])

print("CSV created: 'Site_Specific_Histogram.csv'")