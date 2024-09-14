import torch

NUM_CHANNELS = 3
IMG_SIZE = (120, 160)

def main():
    with torch.inference_mode():
        data = torch.rand((2, 1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1]))
        model = torch.jit.load('/home/feet/Desktop/model-testing-with/Integrated/SSF/balanced/3x160x120/best-r2.pt', 'cpu')
        model.forward(data)

main()