import torch
import onnx
import tensorflow as tf
import torchvision as tv
import onnx_tf
import models.RMEP
import numpy as np

empty = torch.zeros((3,280,280))

model = models.RMEP.Model(10, 3, empty, empty)
model.load_state_dict(torch.load('/home/feet/Documents/LAWN/Visibility-Networks/rewrite2/runs/BestRMEP/best-loss.pt'))
model.eval()
model_name = "bestrmep"

input_shape = (1,3,280,280)
rand_input = torch.from_numpy(np.random.random_sample(input_shape)).to(torch.float32)
print(model(rand_input))

for m in model.modules():
    if 'instancenorm' in m.__class__.__name__.lower():
        m.train(False)

torch.onnx.export(model, torch.randn(input_shape), model_name+'.onnx', opset_version=11)
onnx_model = onnx.load(model_name+'.onnx')
tf_model = onnx_tf.backend.prepare(onnx_model)
tf_model.export_graph(model_name+'.tf')
converter = tf.lite.TFLiteConverter.from_saved_model(model_name+'.tf')
tflite_model = converter.convert()
open(model_name+'.tflite', 'wb').write(tflite_model)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_name+'.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(rand_input, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
