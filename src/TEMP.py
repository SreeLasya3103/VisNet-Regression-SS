import torch
import torchvision.transforms.functional as vfunc

IMG_SIZE = (112, 112)
NUM_CLASSES = 3
NUM_CHANNELS = 3
MASK_SIZE = (22, 22)

target_ratio = IMG_SIZE[0] / IMG_SIZE[1]
ratio = x.size(2) / x.size(3)

if target_ratio >= 1 and ratio > target_ratio:
    x = vfunc.center_crop(x, (x.size(2)/ratio, x.size(3)))
elif target_ratio <= 1 and ratio < target_ratio:
    x = vfunc.center_crop(x, (x.size(2), x.size(3)*ratio))
        
x = vfunc.resize(x, IMG_SIZE, vfunc.InterpolationMode.BICUBIC)



mask_start_h = IMG_SIZE[0]//2 - MASK_SIZE[0]//2
mask_end_h = mask_start_h + MASK_SIZE[0]
mask_start_w = (IMG_SIZE[1]//2+1)//2 - MASK_SIZE[1]//2//2
mask_end_w = mask_start_w + MASK_SIZE[1]//2

fft = x[0,NUM_CHANNELS-1].view((1, 1, *IMG_SIZE))
fft = torch.fft.rfft2(fft)
fft = torch.fft.fftshift(fft)

for i in range(mask_start_h, mask_end_h):
    for j in range(mask_start_w, mask_end_w):
        fft[0,0,i,j] = 0
        
fft = torch.fft.ifftshift(fft)
fft = torch.fft.irfft2(fft, IMG_SIZE)