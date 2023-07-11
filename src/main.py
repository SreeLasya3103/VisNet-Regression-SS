import torch
import tomllib
import sys
sys.path.append("src/datasets")
import FoggyCityscapesDBF as fcs
import FROSI as frosi
sys.path.append('src/models')
import Integrated
import RMEP
import VisNet
    
def main():
    f = open('config.toml', 'rb')
    config = tomllib.load(f)

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
    dataset = None
    if config['dataset'] == 'FCS':
        dataset = fcs.FoggyCityscapesDBF
    elif config['dataset'] == 'FROSI':
        dataset = frosi.FROSI
    
    if config['model'] == 'VISNET':
        VisNet.train_classification(config, use_cuda, dataset)
    elif config['model'] == 'INTEGRATED':
        Integrated.train_classification(config, use_cuda, dataset)
        
            
    
    

if __name__ == '__main__':
    main()