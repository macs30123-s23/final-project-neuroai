import os
import time

import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision

from scipy.stats import ttest_ind

import cornet

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from mpi4py import MPI

Image.warnings.simplefilter('ignore')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
imsize = 224
transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((imsize, imsize)),
                torchvision.transforms.ToTensor(),
                normalize,
            ])

def get_model(pretrained=False):
    map_location = 'cpu'
    model = getattr(cornet, f'cornet_z')
 
    model = model(pretrained=pretrained, map_location=map_location)
    model = model.module  
    return model

model = get_model(pretrained=True)

def read_image(path):
    im = Image.open(path).convert('RGB')
    im = transform(im)
    im = im.unsqueeze(0) 
    return im

def test(layer='V1', sublayer='pool', time_step=0, imsize=224, img_path='test.png'):
    """
    Parameters
    ----------
    layers: str 
        (choose from: V1, V2, V4, IT, decoder)
    sublayer: str 
        (choose from: conv, nonlin, pool, output)
    time_step: int
        which time step to use for storing features, not appliable to Cornet_z
    imsize: int
        resize image to how many pixels, 
        default: 224
    img_path: str
        path to image of interest
        img as batches

    Returns
    -------
    model_feats (numpy array of shape (1, n_neurons))
    """
    model = get_model(pretrained=True)
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize, imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])
    model.eval()

    def _store_feats(layer, inp, output):
        """
        read out inermendiate layer features
        """
        output = output.cpu().numpy()
        _model_feats.append(np.reshape(output, (len(output), -1)))

    m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    model_feats = []
    with torch.no_grad():
        model_feats = []
        im = read_image(img_path)
        _model_feats = [] 
        model(im)
        model_feats.append(_model_feats[time_step])
        model_feats = np.concatenate(model_feats)

    return model_feats

def get_unit_size():
    """
    get the size of each layer
    layers(V1, V2, V4, IT)
    sublayers restricted to pool

    Returns
    -------
    unit_size (dict)
    """
    layers = ['V1', 'V2', 'V4', 'IT']
    unit_size = {}
    for layer in layers:
        output = test(layer=layer)
        unit_size[layer] = output.size
    
    return unit_size

def get_bool_map(layer, job):
    """
    get boolean map for each layer
    layers(V1, V2, V4, IT)
    sublayers restricted to pool

    Parameters
    ----------
    layer: str
        (choose from: V1, V2, V4, IT)
    job: str
        (choose from: grating_hv, grating_tilt, shape)

    Returns
    -------
    bool_map: numpy array of shape (n_neurons)
    """
    start = time.time()
 
    if job != 'shape':
        img_path = os.path.join(stimuli_path, 'grating')
    else:
        img_path = os.path.join(stimuli_path, 'shape')
    
    if job == 'grating_hv':
        img_path_a = os.path.join(img_path, 'h')
        img_path_b = os.path.join(img_path, 'v')
    elif job == 'grating_tilt':
        img_path_a = os.path.join(img_path, 'tilt_45')
        img_path_b = os.path.join(img_path, 'tilt_135')
    else: # job == 'shape'
        img_path_a = os.path.join(img_path, 'circle')
        img_path_b = os.path.join(img_path, 'triangle')

    act_a, act_b = [], []
    act_a = test(layer=layer,
                img_path=img_path_a).tolist()  
    act_b = test(layer=layer,
                img_path=img_path_b).tolist()  
        
    # independent sample t-test
    _, p = ttest_ind(np.array(act_a), np.array(act_b))

    # get boolean map
    bool_map = p < 0.05
        
    return bool_map

def set_parameters():
    imsize = 224
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize, imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])

    unit_size = get_unit_size()

def main():
    start = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    layers = ['V1', 'V2', 'V4', 'IT']
    jobs = ['grating_hv', 'grating_tilt', 'shape']
    
    chunk_size = len(layers) // size
    layers_chunk = layers[rank * chunk_size:(rank + 1) * chunk_size]

    bool_maps = {}

    # Process layers and jobs assigned to this process
    for layer in layers_chunk:
        if layer not in bool_maps:
            bool_maps[layer] = {}

        for job in jobs:
            bool_map = get_bool_map(layer, job)  # Perform your computation here
            bool_maps[layer][job] = bool_map
            print(f'finished {job} in {layer}')

    # Gather bool_maps from all processes onto the root process
    all_bool_maps = comm.gather(bool_maps, root=0)

    # Combine bool_maps on the root process
    if rank == 0:
        combined_bool_maps = {}

        # Merge bool_maps from all processes into combined_bool_maps
        for proc_bool_maps in all_bool_maps:
            for layer, layer_bool_maps in proc_bool_maps.items():
                if layer not in combined_bool_maps:
                    combined_bool_maps[layer] = {}

                for job, bool_map in layer_bool_maps.items():
                    combined_bool_maps[layer][job] = bool_map

    #np.save('bool_maps.npy', bool_maps)
    print(f'finished in {time.time() - start} seconds')

def summary():
    orientation = []
    shape = []
    orientation_shape = []
    for layer in layers:
        print(f'------------------{layer}------------------')
        # orientation selsective 
        orientation_map = np.logical_or(bool_maps[layer]['grating_hv'],
                                        bool_maps[layer]['grating_tilt'])
        orientation.append(np.average(orientation_map))
        # shape selective
        shape_map = bool_maps[layer]['shape']
        shape.append(np.average(shape_map))

        # orientation and shape selective
        orientation_shape_map = np.logical_and(orientation_map, shape_map)
        orientation_shape.append(np.average(orientation_shape_map))

        print(f'layer {layer}, orientation selective: {np.average(orientation_map)}')
        print(f'layer {layer}, shape selective: {np.average(shape_map)}')
        print(f'layer {layer}, orientation and shape selective: {np.average(orientation_shape_map)}')   

    
    # construct dataframe for visualization
    df_orientation = pd.DataFrame({'layer': layers, 
                                   'proportion': orientation})
    df_shape = pd.DataFrame({'layer': layers, 
                             'proportion': shape})
    df_orientation_shape = pd.DataFrame({'layer': layers, 
                                         'proportion': orientation_shape})

def sub_plot(df, title, filename):
    sns.barplot(x='layer', y='proportion', data=df)
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Proportion')
    plt.savefig(os.path.join('figures', filename), dpi=300)
    plt.close()

def plot():
    sub_plot(df_orientation, 
             'Orientation Selective', 
             'Orientation Selective.png')
    sub_plot(df_shape, 
             'Shape Selective', 
             'Shape Selective.png')
    sub_plot(df_orientation_shape, 
             'Orientation and Shape Selective', 
             'Conjunction.png')
            


if __name__ == '__main__':
    stimuli_path = '/Users/anmin/Documents/Courses/2023Spring\
/Large_scale_computing\
/stimuli/'
    
    set_parameters()
    main()
    plot()