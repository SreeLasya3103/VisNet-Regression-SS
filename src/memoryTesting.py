import torch

NUM_CHANNELS = 3
IMG_SIZE = (200, 200)


def main():
    with torch.inference_mode():
        data = torch.rand((3, 1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1]))
        model = torch.jit.load('/home/feet/Repos/Visibility-Networks/src/models/VisNet-3x200x200-3-Q.pt', 'cpu')
        model.forward(data)

main()