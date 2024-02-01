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
import VisNetReduced

IMG_SIZE = (120, 160)
NUM_CLASSES = 1
NUM_CHANNELS = 3

with torch.inference_mode():
   sample = torch.rand((1, NUM_CHANNELS, IMG_SIZE[0], IMG_SIZE[1]))

   model = torch.jit.load('/home/feet/Desktop/model-testing-with/RMEP/SSF/3x160x120/best-r2.pt', map_location=torch.device('cpu'))

   state = model.state_dict()

   model = rmep.RMEP()
   model(sample)
   model.load_state_dict(state)

   model_dynamic_quantized = torch.quantization.quantize_dynamic(
   model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

   model_dynamic_quantized = torch.jit.script(model_dynamic_quantized)
   model_dynamic_quantized.save('RMEP-SSF-3x160x120-1-Q.pt')