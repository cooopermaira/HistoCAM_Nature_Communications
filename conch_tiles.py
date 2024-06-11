# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:25:41 2024

@author: Max Cooper
"""

from PIL import Image
from itertools import product
import os

filename = "breast_20x-05162024121810-0.png";
dir_in = r"D:\conch_test";
dir_out = r"D:\conch_test\tiles";
d = 256; 

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)
        
tile(filename, dir_in, dir_out, d)