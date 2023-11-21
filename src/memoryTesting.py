import torch

NUM_CHANNELS = 1
IMG_SIZE = (112, 112)


def main():
    with torch.inference_mode():
        data = torch.rand((3, 1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1]))
        model = torch.jit.load('/home/feet/Repos/Visibility-Networks/src/models/VisNet_Reduced-1x112x112-3.pt', 'cpu')
        model.forward(data)

main()