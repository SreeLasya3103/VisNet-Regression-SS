import torch
from memory_profiler import profile, memory_usage

NUM_CHANNELS = 3
IMG_SIZE = (120, 160)

@profile
def main():
    with torch.inference_mode():
        data = torch.rand((3, 1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1]))
        model = torch.jit.load('/home/feet/Repos/Visibility-Networks/example/trained-visnet-reduced-3x160x120.pt', 'cpu')
        mem = memory_usage(proc=(model.forward, [data]), max_usage=True, include_children=True)
        print(mem)
