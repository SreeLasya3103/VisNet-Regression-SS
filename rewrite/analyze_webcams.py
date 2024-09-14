import torch
import torch.nn as nn
import dsets
import dsets.Webcams
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime
import pprint

class_names = ('1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
dset_dir = '/home/feet/Documents/LAWN/datasets/Webcams'

info_list = []

for subdir, dirs, files in os.walk(dset_dir):
    dir_name = os.path.basename(subdir)
    if dir_name == os.path.basename(dset_dir):
        continue
    
    try:
        timestamp = datetime.strptime(dir_name, '%Y-%m-%d-%H-%M-%S')
    except:
        timestamp = 'N/A'
    
    for file in files:
        name = file[0:-4]
        site, ornt, vis = tuple(name.split('_'))
        
        vis = vis.replace('-', '.')
        floatvis = float(vis[3:-2])
        if floatvis > 10.0:
            vis = "VIS10mi"
        
        info_list += [(site, ornt, vis, timestamp)]

print(len(info_list))
    

site_counts = dict()
for info in info_list:
    if info[0] not in site_counts:
        site_counts[info[0]] = 0
    site_counts[info[0]] += 1
    
# pprint.pprint(site_counts)
# print(len(site_counts))
    
site_vis_counts = dict()
for info in info_list:
    if info[0] not in site_vis_counts:
        site_vis_counts[info[0]] = dict()
    if info[2] not in site_vis_counts[info[0]]:
        site_vis_counts[info[0]][info[2]] = 0
    site_vis_counts[info[0]][info[2]] += 1
    
# pprint.pprint(site_vis_counts)

vis_site_counts = dict()
for info in info_list:
    if info[2] not in vis_site_counts:
        vis_site_counts[info[2]] = dict()
    if info[0] not in vis_site_counts[info[2]]:
        vis_site_counts[info[2]][info[0]] = 0
    vis_site_counts[info[2]][info[0]] += 1
    
# pprint.pprint(vis_site_counts)


vis_difsites_counts = dict()
vis_difsites = dict()
for info in info_list:
    if info[2] not in vis_difsites:
        vis_difsites_counts[info[2]] = 0
        vis_difsites[info[2]] = set()
    if info[0] not in vis_difsites[info[2]]:
        vis_difsites_counts[info[2]] += 1
        vis_difsites[info[2]].add(info[0])
        
site_or_counts = dict()
difvis_counts = dict()
difviss = dict()
for info in info_list:
    if (info[0], info[1]) not in site_or_counts:
        site_or_counts[(info[0], info[1])] = 0
        difvis_counts[(info[0], info[1])] = 1
        difviss[(info[0], info[1])] = set()
        difviss[(info[0], info[1])].add(info[2])
    else:
        site_or_counts[(info[0], info[1])] += 1
        if info[2] not in difviss[(info[0], info[1])]:
            difvis_counts[(info[0], info[1])] += 1
            difviss[(info[0], info[1])].add(info[2])
        
        
# print('Number of sites:', len(site_counts))
# print('Images per site:')
# pprint.pprint(site_counts)
# print('\nVis classes of each site:')
# pprint.pprint(site_vis_counts)
# print('\nNumber of different sites for each vis class:')
# pprint.pprint(vis_difsites_counts)
difvis_counts = sorted(difvis_counts.items(), key=lambda x:x[1])
pprint.pprint(difvis_counts)
    


# 1.0: 135, 2.0: 200, 3.0: 263, 4.0: 254, 5.0: 313, 6.0: 382, 7.0: 450, 8.0: 407, 9.0: 7648, 10.0: 2193
# SITE41_ORNT335
# SITE41_ORNT25
# SITE41_ORNT235 !!!!!!!!!!!!!!!!!!!!!!!!
# SITE41_ORNT170

# SITE203_ORNT220
# SITE203_ORNT70
# SITE203_ORNT155