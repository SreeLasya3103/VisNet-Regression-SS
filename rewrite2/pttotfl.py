import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import torchvision as tv
import models.RMEP

INPUT_SHAPE = (1,3,280,280)
empty = torch.zeros(INPUT_SHAPE)

model_name = "bestrmep"

class WithSigmoid(nn.Module):
    def __init__(self):
        super(WithSigmoid, self).__init__()

        self.model = tv.models.resnet34(num_classes=1)
    
    def forward(self, x):
        return nn.functional.sigmoid(self.model(x))
    
class WithSoftmax(nn.Module):
    def __init__(self):
        super(WithSoftmax, self).__init__()

        self.model = models.RMEP.Model(10, 3, empty[0], empty[0])

    def forward(self, x):
        return nn.functional.softmax(self.model(x), 1)

model = models.RMEP.Model(10, 3, empty[0], empty[0])
model.load_state_dict(torch.load('/home/feet/Documents/LAWN/Visibility-Networks/rewrite2/runs/BestRMEP/best-loss.pt', weights_only=False, map_location=torch.device('cpu')))
model.eval()
for m in model.modules():
    m.train(False)
onnx_filename = model_name + ".onnx"

rand_in = torch.randn(INPUT_SHAPE, dtype=torch.float)
preconv = model(rand_in)
torch.onnx.export(model, rand_in, model_name, input_names=['input'], output_names=['output'], opset_version=11)

onnx_model = onnx.load(model_name)
tf_rep = prepare(onnx_model)

tf_model_dir = "./tf_model"
tf_rep.export_graph(tf_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(model_name+ ".tflite", "wb") as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_name+'.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(rand_in, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
postconv = interpreter.get_tensor(output_details[0]['index'])

# print(preconv)
# print(postconv)