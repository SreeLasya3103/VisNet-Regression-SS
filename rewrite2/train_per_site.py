import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from image_cropping import get_resize_crop_fn
from models.VisNet import get_tf_function
from train_val import train_reg
from models import VisNet
from dsets.Webcams import Webcams_reg
from glob import glob
from tqdm import tqdm

# Dataset paths
dataset_path = "D:\\Research - Lasya\\VEIA"
all_jpgs = glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
all_sites = sorted(set([os.path.basename(f).split('_')[0] for f in all_jpgs]))

# Output paths
output_dir = "D:\\Research - Lasya\\VisNet-Regression-SS\\rewrite2"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "Final_Details.csv")

# Transforms
resize_fn = get_resize_crop_fn((280, 280))
visnet_tf = get_tf_function()
transform = lambda x, augment=False: visnet_tf(resize_fn(x))

# Store results
all_results = []

for site in tqdm(all_sites, desc="Training all sites"):
    print(f"\n=== Training model for {site} ===")

    writer = SummaryWriter(log_dir=f"runs/per_site/{site}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Load dataset for one site
    dset = Webcams_reg(dataset_path, transformer=transform, site_filter=[site])

    if len(dset) < 10:
        print(f"[!] Skipping {site}: only {len(dset)} samples.")
        continue

    # Train/Val/Test split
    num_train = int(0.8 * len(dset))
    num_val = int(0.1 * len(dset))
    train, val, test = torch.utils.data.random_split(dset, [num_train, num_val, len(dset) - num_train - num_val])

    loaders = (
        DataLoader(train, batch_size=8, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=8),
        DataLoader(test, batch_size=8),
    )

    # Choose the model here
    model = VisNet.Model(num_classes=1, num_channels=3, mean=torch.zeros((3,)), std=torch.ones((3,)))
    # from models import RMEP; model = RMEP.Model(...)  ← if running RMEP
    model = model.cuda() if torch.cuda.is_available() else model
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)

    # Train & evaluate
    test_data = train_reg(
        loaders=loaders,
        model=model,
        optimizer=optimizer,
        loss_fn=nn.SmoothL1Loss(),
        epochs=15,
        use_cuda=torch.cuda.is_available(),
        subbatch_count=4,
        output_fn=None,
        labels_fn=None,
        writer=writer,
        transform=transform
    )

    # Save result
    if test_data:
        all_results.append({
            'site_id': site,
            'r2': test_data['r2'],
            'mse': test_data['mse'],
            'log_dir': writer.get_logdir()
        })

    # Save CSV incrementally
    df = pd.DataFrame(all_results)
    df.drop(columns='confusion_matrix', errors='ignore').to_csv(csv_path, index=False)
    print(f"[✓] Partial save after {site} to: {csv_path}")

print(f"\n✅ Final results saved to: {csv_path}")
