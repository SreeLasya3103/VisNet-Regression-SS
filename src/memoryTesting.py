import torch

NUM_CHANNELS = 3
IMG_SIZE = (200, 200)

def main():
    with torch.inference_mode():
        data = torch.rand((3, 1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1]))
        model = torch.jit.load('/home/feet/Desktop/model-testing-with/VisNet/FCS/3x200x200/best-acc.pt', 'cpu')
        model.forward(data)

main()