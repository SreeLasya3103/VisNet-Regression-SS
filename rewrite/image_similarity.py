import os
import torch.utils.data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import sortedcontainers as sc
from dsets.Webcams import Webcams_cls
import sklearn_extra.cluster as skc
import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.transforms.functional as tvtf
import torch
import pickle
import numpy as np
import torch.utils
from torcheval.metrics.functional import r2_score, multiclass_confusion_matrix
import seaborn as sn
import pandas as pd
from progress.bar import Bar


# LIMITS = {3.0:120, 4.0:120, 5.0:120, 6.0:120, 7.0:120, 8.0:120, 9.0:120, 10.0:120}
LIMITS = {1.0:0, 1.25:0, 1.5:0, 1.75:0, 2.0:0, 2.25:0, 2.5:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0, 10.0:0}
OTHER_LIMITS = {1.00:500, 1.25:500, 1.5:500, 1.75:500, 2.0:500, 2.25:500, 2.5:500, 3.0:500, 4.0:500, 5.0:500, 6.0:500, 7.0:500, 8.0:500, 9.0:500, 10.0:500}
CLASSES = 3

# for image_path in dset.files:
#     drctry = os.path.dirname(image_path).replace('/home/feet/Documents/LAWN/datasets/WebcamsAlreadyCropped', '/home/feet/Documents/LAWN/low-quality-image-detection-main/dset')
#     os.system('mkdir ' + drctry)
#     os.system('cp ' + image_path + ' ' + drctry)

# DataLoader(dset, 32, False, num_workers=4)

# Load the OpenAI CLIP Model
# print('Loading CLIP Model...')
# dset = Webcams_cls_10('/home/feet/Documents/LAWN/datasets/WebcamsAlreadyCropped', limits={1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0, 10.0:0})

model = SentenceTransformer('clip-ViT-B-32')

class_avg = []
val_set = []

dset = Webcams_cls('/home/feet/Documents/LAWN/datasets/Webcams', limits=OTHER_LIMITS)

for i in range(CLASSES):
    lim = LIMITS.copy()
    # lim[float(i+1)] = 200
    # if i == 0:
    #     lim[float(i+1.25)] = 200
    # elif i == 1:
    #     lim[float(i+0.75)] = lim[float(i+1.25)] = 200
        
    if i <= 7:
        lim[1.0] = lim[1.25] = lim[1.5] = lim[1.75] = lim[2.0] = lim[2.25] = lim[2.5] = lim[3.0] = 200
    elif i <= 11:
        lim[4.0] = lim[5.0] = lim[6.0] = lim[7.0] = 200
    else:
        lim[8.0] = lim[9.0] = lim[10.0] = 200
        
    class_set = Webcams_cls(dset.files, limits=lim)
    splits = np.split(np.array(class_set.files), [int(len(class_set.files)*0.75)])
    image_names = splits[0].tolist()
    val_set += splits[1].tolist()
    train_set = Webcams_cls(image_names, transform=lambda x, _: tvtf.resize(x, (309,470)))
    dloader = torch.utils.data.DataLoader(train_set)
    
    encoded_image = model.encode([tvtf.to_pil_image(tvtf.convert_image_dtype(img[0], torch.uint8)) for img, _ in dloader], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
    class_avg.append(torch.sum(encoded_image, 0) / len(encoded_image))

val_set = Webcams_cls(val_set, transform=lambda x, _: tvtf.resize(x, (309,470)))

pred_indices = []
target_indices = []

correct = 0

bar = Bar()
bar.max = len(val_set)

for img, label in val_set:
    scores = []
    for i in range(CLASSES):
        # comp_image = model.encode(Image.open(img), convert_to_tensor=True)
        comp_image = model.encode(tvtf.to_pil_image(tvtf.convert_image_dtype(img[0], torch.uint8)), convert_to_tensor=True)
        encoded_image = torch.stack((class_avg[i], comp_image))
        

        # Now we run the clustering algorithm. This function compares images aganist 
        # all other images and returns a list with the pairs that have the highest 
        # cosine similarity score
        processed_images = util.paraphrase_mining_embeddings(encoded_image, max_pairs=75000, top_k=100000)
        # NUM_SIMILAR_IMAGES = 1000000

        # dm = [ [0.0]*len(image_names) for _ in range(len(image_names))]
        
        
        for score, image_id1, image_id2 in processed_images:
            # dm[image_id1][image_id2] = score
            # dm[image_id2][image_id1] = score
            # print(image_names[image_id1], image_names[image_id2], score, sep=',')
            # print(i+1, score, sep=',')
            scores.append(score)

    max_score = np.argmax(scores).item()
    class_index = label.argmax().item()
    
    if class_index <= 7:
        class_index = 0
    elif class_index <= 11:
        class_index = 1
    else:
        class_index = 2
    
    # if class_index == 2 or class_index == 6:
    #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # elif class_index <= 1:
    #     class_index = 0
    # elif class_index <= 5:
    #     class_index = 1
    # else:
    #     class_index -= 5
    
    # match float_value:
    #     case 1.0:
    #         class_index = 0
    #     case 1.25:
    #         class_index = 0
    #     case 1.75:
    #         class_index = 1
    #     case 2.0:
    #         class_index = 1
    #     case 2.25:
    #         class_index = 1
    #     case 3.0:
    #         class_index = 2
    #     case 4.0:
    #         class_index = 3
    #     case 5.0:
    #         class_index = 4
    #     case 6.0:
    #         class_index = 5
    #     case 7.0:
    #         class_index = 6
    #     case 8.0:
    #         class_index = 7
    #     case 9.0:
    #         class_index = 8
    #     case 10.0:
    #         class_index = 9
    
    if max_score == class_index:
        correct += 1
    
    pred_indices.append(max_score)
    target_indices.append(class_index)
    
    bar.next()

conf_mat = multiclass_confusion_matrix(torch.Tensor(pred_indices).to(torch.int64), torch.Tensor(target_indices).to(torch.int64), CLASSES, normalize='true')
# vcm = pd.DataFrame(conf_mat, index=["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"], columns=["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"])
vcm = pd.DataFrame(conf_mat, index=["1-3", "4-7", "8-10"], columns=["1-3", "4-7", "8-10"])
plot = sn.heatmap(vcm, annot=True, vmin=0.0, vmax=1.0)
plot.set_xlabel('Predicted Value')
plot.set_ylabel('True Value')
plt.savefig("simconfmat.png")

print("accuracy: " + str(correct/len(val_set)))



# avsims = [0.0 for _ in range(len(image_names))]

# for i in range(0, len(image_names)):
#     avsims[i] = sum(dm[i]) / (len(image_names)-1)
#     if(image_names[i] == "/home/feet/Documents/LAWN/datasets/WebcamsAlreadyCropped/2024-05-07-15-42-07/SITE561_ORNT60_VIS3mi.png"):
#         print(image_names[i]+','+str(avsims[i]))

# k = 10

# meat = skc.KMedoids(k, 'precomputed', 'pam').fit_predict(dm)

# classes = [[] for _ in range(k)]
# os.system('rm -r ./kmedoid_classes/*')
# for i in range(k):
#     os.system('mkdir ./kmedoid_classes/' + str(i))

# for i in range(len(image_names)):
#     classes[meat[i]] += [ os.path.basename(image_names[i][0:-4]) ]
#     os.system('cp ' + image_names[i] + ' ./kmedoid_classes/' + str(meat[i]) + '/')
    
# averages = [0.0 for _ in range(k)]
# num_sites = [0 for _ in range(k)]
# site_perc = [0.0 for _ in range(k)]
# for i in range(k):
#     sites = dict()
#     for img in classes[i]:
#         site, ornt, vis = tuple(img.split('_'))
#         vis = vis.replace('-', '.')
#         floatvis = float(vis[3:-2])
#         if floatvis > 10.0:
#             vis = "VIS10mi"
#             floatvis = 10.0

#         if site+ornt in sites:
#             sites[site+ornt] += 1
#         else:
#             sites[site+ornt] = 1

#         averages[i] += floatvis
    
#     num_sites[i] = len(sites)
#     for count in sites.values():
#         site_perc[i] = max(site_perc[i], count/len(classes[i]))
    
#     averages[i] /= len(classes[i])

# sorted_indices = sorted(range(0, k), key=lambda i:averages[i])

# averages = [averages[i] for i in sorted_indices]
# num_sites = [num_sites[i] for i in sorted_indices]
# site_perc = [site_perc[i] for i in sorted_indices]

# print('Classes')
# for i in sorted_indices:
#     print('{0: <7}'.format(i), end=' ')
# print('\nAverage visibility')
# for avg in averages:
#     print('{0: <7}'.format('%.2f' % avg), end=' ')
# print('\nNumber of sites')
# for num in num_sites:
#     print('{0: <7}'.format(num), end=' ')
# print('\nHighest percentage from one site')
# for perc in site_perc:
#     print('{0: <7}'.format('%.2f' % perc), end=' ')
# print('')

# # =================
# # DUPLICATES
# # =================
# print('Finding duplicate images...')
# # Filter list for duplicates. Results are triplets (score, image_id1, image_id2) and is scorted in decreasing order
# # A duplicate image will have a score of 1.00
# # It may be 0.9999 due to lossy image compression (.jpg)
# duplicates = [image for image in processed_images if image[0] >= 0.999]

# # Output the top X duplicate images
# for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
#     print("\nScore: {:.3f}%".format(score * 100))
#     print(image_names[image_id1])
#     print(image_names[image_id2])

# =================
# NEAR DUPLICATES
# =================
# print('Finding near duplicate images...')
# Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
# you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
# A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
# duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.
# threshold = 0.50
# near_duplicates = [image for image in processed_images if 0.999 > image[0] >= threshold]

# samples = sc.SortedKeyList(key=lambda x: x[0])

# for score, image_id1, image_id2 in near_duplicates:
#     # print("Score: {:.3f}%".format(score * 100), '   ', image_names[image_id1], '    ', image_names[image_id2])
#     # print(image_names[image_id1])
#     # print(image_names[image_id2])

    
#     name = os.path.basename(image_names[image_id1])[0:-4]
#     site, ornt, vis = tuple(name.split('_')) 
#     vis = vis.replace('-', '.')
#     floatvis = float(vis[3:-2])
#     if floatvis > 10.0:
#         vis = "VIS10mi"
    
#     img_info1 = (site, ornt, vis)
    
#     name = os.path.basename(image_names[image_id2])[0:-4]
#     site, ornt, vis = tuple(name.split('_')) 
#     vis = vis.replace('-', '.')
#     floatvis = float(vis[3:-2])
#     if floatvis > 10.0:
#         vis = "VIS10mi"
    
#     img_info2 = (site, ornt, vis)
    
#     # if img_info1[0:2] == img_info2[0:2]:
#     #     continue
    
#     # if img_info1[2]==img_info2[2]:
#     #     samevis = 1.0
#     # else:
#     #     samevis = 0.0
    
#     # samples.add((score, samevis))
    


