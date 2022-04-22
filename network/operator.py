import torch
import torch.nn.functional as F

def normalize_coords(coords: torch.Tensor, h, w):
    """
    normalzie coords to [-1,1]
    @param coords:
    @param h:
    @param w:
    @return:
    """
    coords = torch.clone(coords)
    coords = coords + 0.5
    coords[...,0] = coords[...,0]/w
    coords[...,1] = coords[...,1]/h
    coords = (coords - 0.5)*2
    return coords

def denormalize_coords(coords, h, w):
    coords = coords / 2 + 0.5
    coords[...,0] *= w
    coords[...,1] *= h
    coords -= 0.5
    return coords