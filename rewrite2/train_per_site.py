import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from image_cropping import get_resize_crop_fn
from models.VisNet import get_tf_function
from train_val import train_cls
from models import VisNet
from dsets.Webcams import Webcams_cls_10
from glob import glob
from tqdm import tqdm

dataset_path = "C:\\Users\\sm380923\\Desktop\\Research\\good"
all_pngs = glob(os.path.join(dataset_path, "**", "*.png"), recursive=True)
all_sites = sorted(set([os.path.basename(f).split('_')[0] for f in all_pngs]))

all_results = []

output_dir = "C:\\Users\\sm380923\\Desktop\\Research\\Visibility-Networks\\rewrite2"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "All_Sites_Metrics.csv")

resize_fn = get_resize_crop_fn((310, 470))
visnet_tf = get_tf_function()
transform = lambda x, augment=False: visnet_tf(resize_fn(x))

results = []

for site in tqdm(all_sites, desc="Training all sites"):
    print(f"\n=== Training model for {site} ===")

    writer = SummaryWriter(log_dir=f"runs/per_site/{site}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    dset = Webcams_cls_10(dataset_path, transform=transform, site_filter=[site])

    #if len(dset) < 50:
        #print(f"[!] Skipping {site}: Not enough data")
        #continue

    num_train = int(0.8 * len(dset))
    num_val = int(0.1 * len(dset))
    train, val, test = torch.utils.data.random_split(dset, [num_train, num_val, len(dset) - num_train - num_val])

    loaders = (
        DataLoader(train, batch_size=8, shuffle=True),
        DataLoader(val, batch_size=8),
        DataLoader(test, batch_size=8),
    )

    model = VisNet.Model(num_classes=10, num_channels=3, mean=torch.zeros((3,)), std=torch.ones((3,)))
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)

    test_data  = train_cls(
        loaders=loaders,
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        epochs=15,
        use_cuda=torch.cuda.is_available(),
        subbatch_count=1,
        class_names=[str(i) for i in range(1, 11)],
        output_fn=None,
        labels_fn=None,
        writer=writer,
        transform=transform
    )

    if test_data and 'site_metrics' in test_data:
        for row in test_data['site_metrics']:
            row['site_id'] = site
            row['log_dir'] = writer.get_logdir()
            all_results.append(row)

    df = pd.DataFrame(all_results)
    df.drop(columns='confusion_matrix', errors='ignore').to_csv(csv_path, index=False)
    print(f"Partial save after {site} to: {csv_path}")
    print(f"\n Final results saved to: {csv_path}")



