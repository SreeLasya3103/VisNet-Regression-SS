import ai_edge_torch
import torch
import numpy as np
import torchvision as tv

empty = torch.zeros((3,280,280))

model = tv.models.resnet34(num_classes=1)
model.load_state_dict(torch.load('/home/feet/Documents/LAWN/Visibility-Networks/rewrite/goodbad-bestloss.pt', map_location=torch.device('cpu')))
model.eval()
model_name = "goodbadresnet"

input_shape = (1,3,280,280)
rand_input = torch.from_numpy(np.random.random_sample(input_shape)).to(torch.float32)

edge_model = ai_edge_torch.convert(model, rand_input)

print(model(rand_input))
print(edge_model(rand_input))