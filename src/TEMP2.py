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

with torch.inference_mode():
    model = torch.jit.load('/home/feet/Repos/Visibility-Networks/src/models/RMEP-3x160x120-1.pt', torch.device('cpu'))
    model.eval()

state = model.state_dict()

model = rmep.RMEP()
model.load_state_dict(state)

torch.save(model, "./trained-rmep-3x160x120-NOT-torchscript.pt")