import numpy as np
import matplotlib.pyplot as plt
import random as rand
import open3d as o3d
import time
from utility import *
from epnp import EPnP
from tqdm import trange






Tr_random = compute_T(
    np.random.uniform(-np.pi, np.pi),   # x-rotation
    np.random.uniform(-np.pi, np.pi),   # y-rotation
    np.random.uniform(-np.pi, np.pi),   # z-rotation 
    np.random.uniform(-0.5,0.5),        # x-translation
    np.random.uniform(-0.5,0.5),        # x-translation
    np.random.uniform(-0.5,0.5)*3+10)   # x-translation