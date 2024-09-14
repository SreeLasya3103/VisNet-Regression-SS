import torch
import tomli
import sys
import os
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
import FoggyCityscapesDBF as fcs
import FROSI as frosi
import SSF as ssf
import SSF_YCbCr as ssf_YCbCr
import AllSets
import Jacobs
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import Integrated
import RMEP as rmep
import VisNet
import onnx

model = torch.load('/home/feet/Desktop/model-testing-with/RMEP/SSF/NewMaxPool/3x160x120/best-r2.pt', torch.device('cpu'))
model.eval()

x = torch.rand((1, 3, 120, 160))

torch.onnx.export(model, x, "rmep-3x160x120-1.onnx", export_params=True, opset_version=10,
                  do_constant_folding=True, input_names = ['input'],
                  output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'},
                                                         'output' : {0 : 'batch_size'}})

