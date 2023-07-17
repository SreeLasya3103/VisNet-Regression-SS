import torch
import tomli
import sys
import os
ROOT_DIR = os.path.dirname(__file__)
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
import FoggyCityscapesDBF as fcs
import FROSI as frosi
import SSF as ssf
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import Integrated
import RMEP as rmep
import VisNet
    
def main():
    f = open('config.toml', 'rb')
    config = tomli.load(f)

    try_cuda = True
    if 'try_cuda' in config:
        try_cuda = config['try_cuda']
    
    use_cuda = False
    if try_cuda:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('CUDA available. Using GPU...')
        else:
            print('CUDA unavailable. Using CPU...')
    else:
        print('Using CPU...')
        
    if config['mode'] == 'TRAIN':
        train(config, use_cuda)
    elif config['mode'] == 'TEST':
        print('TESTING')
    elif config['mode'] == 'VALIDATE':
        print('VALIDATING')
    else:
        print('No mode specified!')
    
def train(config, use_cuda):
    model_module = None
    
    if config['model'] == 'VISNET':
        model_module = VisNet
    elif config['model'] == 'INTEGRATED':
        model_module = Integrated
    elif config['model'] == 'RMEP':
        model_module = rmep
        
    dataset = None
    if config['dataset'] == 'FCS':
        dataset = fcs.FoggyCityscapesDBF
        model_module.train_classification(config, use_cuda, dataset)
    elif config['dataset'] == 'FROSI':
        dataset = frosi.FROSI
        model_module.train_classification(config, use_cuda, dataset)
    elif config['dataset'] == 'SSF':
        dataset = ssf.SSF
        model_module.train_regression(config, use_cuda, dataset)
    elif config['dataset'] == 'OTHER':
        dataset = None
        model_module.train_regression(config, use_cuda, dataset)
    
        
            
    
    

if __name__ == '__main__':
    main()
