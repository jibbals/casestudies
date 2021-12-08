# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:35:56 2021

@author: jgreensl
"""


import matplotlib.pyplot as plt
import numpy as np

from mayavi import mlab

from timeit import default_timer as timer

from utilities import utils,fio,constants

mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))
mlab.clf()

# The position of the atoms
atoms_x = np.array([2.9, 2.9, 3.8]) * 40 / 5.5
atoms_y = np.array([3.0, 3.0, 3.0]) * 40 / 5.5
atoms_z = np.array([3.8, 2.9, 2.7]) * 40 / 5.5

O = mlab.points3d(atoms_x[1:-1], atoms_y[1:-1], atoms_z[1:-1],
                  scale_factor=3,
                  resolution=20,
                  color=(1, 0, 0),
                  scale_mode='none')

H1 = mlab.points3d(atoms_x[:1], atoms_y[:1], atoms_z[:1],
                   scale_factor=2,
                   resolution=20,
                   color=(1, 1, 1),
                   scale_mode='none')

H2 = mlab.points3d(atoms_x[-1:], atoms_y[-1:], atoms_z[-1:],
                   scale_factor=2,
                   resolution=20,
                   color=(1, 1, 1),
                   scale_mode='none')

# The bounds between the atoms, we use the scalar information to give
# color
mlab.plot3d(atoms_x, atoms_y, atoms_z, [1, 2, 1],
            tube_radius=0.4, colormap='Reds')


mlab.view(132, 54, 45, [21, 20, 21.5])

mlab.show()