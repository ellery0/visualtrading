#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:44:50 2020

@author: fuy
"""

from keras.models import load_model
from keras.utils import plot_model 
model_name=['embedding_model','embedding','model']
for name in model_name:
    model = load_model('%s.h5'%name)
    plot_model(model,to_file='%s.png'%name,show_shapes=False)














