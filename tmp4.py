import torch

def createExampleData(num_classes, count):
    outputs = torch.stack([torch.nn.functional.softmax(torch.rand((num_classes)), dim=0) for _ in range(count)], 0)
    labels = torch.cat([torch.nn.functional.one_hot(torch.randint(0, num_classes, (1,)), num_classes) for _ in range(count)], 0)

    return (outputs, labels)

print(createExampleData(4, 10)[0].size())