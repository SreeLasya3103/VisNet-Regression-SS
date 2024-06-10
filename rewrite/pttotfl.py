import torch
import onnx
import tensorflow as tf
import onnx_tf
import models.RMEP

empty = torch.zeroes((3,200,200))

model = models.RMEP.Model(15,3,empty,empty)
model.load_state_dict('testm.pt')
model.eval()
model_name = ":^)"

input_shape = (1,3,200,200)

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