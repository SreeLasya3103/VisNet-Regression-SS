import tomli
from os import path as os_path
from sys import path as sys_path
from sys import argv
import torch
import torchvision.transforms.functional as vfunc
from PIL import Image
from torchvision import transforms
from matplotlib.colors import LinearSegmentedColormap
from picamera import PiCamera
from time import sleep

ROOT_DIR = os_path.dirname(__file__)
sys_path.append(os_path.join(ROOT_DIR, 'models'))

def resize_crop(img, img_dim):
    target_ratio = img_dim[0] / img_dim[1]
    ratio = img.size(1) / img.size(2)
    
    if ratio > target_ratio:
        img = vfunc.center_crop(img, (round(img.size(2)*target_ratio), img.size(2)))
    elif ratio < target_ratio:
        img = vfunc.center_crop(img, (img.size(1), round(img.size(1)/target_ratio)))
    
    img = vfunc.resize(img, img_dim, vfunc.InterpolationMode.BICUBIC, antialias=False)

    return img

def highpass_filter(img, mask_dim):
    orig_dim = (img.size(1), img.size(2))
    img = torch.fft.rfft2(img)
    img = torch.fft.fftshift(img)
    
    h_start = img.size(1)//2 - mask_dim[0]//2
    w_start = img.size(2)//2 - mask_dim[0]//2//2

    for i in range(h_start, h_start+mask_dim[0]):
        for j in range(w_start, w_start+mask_dim[1]//2):
            img[0][i][j] = 0

    img = torch.fft.ifftshift(img)    
    img = torch.fft.irfft2(img, orig_dim)
    
    return img

f = open('config.toml', 'rb')
config = tomli.load(f)

img_dim = config['imgDim']

with torch.inference_mode():
    model = torch.jit.load(config['modelPath'], torch.device('cpu'))
    camera = PiCamera()
    sleep(2)

    while True:
        camera.capture('./img.png')

        orig = Image.open('./img.png').convert('RGB')
        orig = transforms.PILToTensor()(orig)
        orig = resize_crop(orig, img_dim) / 255
        pc = None
        fft = None

        if config['model'] == 'VISNET':
            pc_cmap = LinearSegmentedColormap.from_list('', ['#000000', '#3F003F', '#7E007E',
                                                            '#4300BD', '#0300FD', '#003F82',
                                                            '#007D05', '#7CBE00', '#FBFE00',
                                                            '#FF7F00', '#FF0500'])
            
            pc = orig[2].detach().clone()
            pc = pc.view(1, *img_dim)
            pc = pc_cmap(pc)
            pc = torch.from_numpy(pc).permute((0,3,1,2))
            pc = torch.cat((pc[0][0], pc[0][1], pc[0][2])).view(-1, *img_dim)

            fft = orig.detach().clone()
            fft = fft.view(1, *img_dim)
            fft = highpass_filter(fft, config['maskDim'])
            fft = torch.clamp(fft, 0.0, 1.0)
            fft = pc_cmap(fft)
            fft = torch.from_numpy(fft).permute((0, 3, 1, 2))
            fft = torch.cat((fft[0][0], fft[0][1], fft[0][2])).view(-1, *img_dim)
        elif config['model'] == 'INTEGRATED':
            pc_cmap = LinearSegmentedColormap.from_list('', ['#0000ff', '#00ff00', '#ff0000', '#0000ff'])
            
            pc = transforms.Grayscale()(orig)
            pc = pc.view(1, *img_dim)
            pc = pc_cmap(pc)
            pc = torch.from_numpy(pc).permute((0,3,1,2))
            pc = torch.cat((pc[0][0], pc[0][1], pc[0][2])).view(-1, *img_dim)
        elif config['model'] != 'RMEP':
            print('Not a valid model type')
            exit()

        data = orig.view((1, 1, -1, *img_dim))
        if pc is not None:
            data = torch.cat((data, (pc.view((1, 1, -1, *img_dim)))))
        if fft is not None:
            data = torch.cat((data, (fft.view((1, 1, -1, *img_dim)))))

        if config['model'] == 'RMEP':
            data = data[0]

        output = model(data)
        print(output)
    