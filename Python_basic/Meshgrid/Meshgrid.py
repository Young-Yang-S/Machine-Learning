# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:38:18 2020

@author: daiya
"""

# The numpy.meshgrid function is used to create a rectangular grid out of two given 
# one-dimensional arrays representing the Cartesian indexing or Matrix indexing.
# then we can use contour to draw the plot with different darkness of color to represent
# the different function values

# It is popular to use as tool to draw boundary in ML

# Example 1
import matplotlib.pyplot as plt
import numpy as np
x = [0, 1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6, 7, 8,9]
xx, yy = np.meshgrid(x, y)
# after using meshgrid, we get 8*6 matrics, where 8 is the number of y row, 6 is the number
# of column of x, y is the same.
# after doing this we can get every grid point of a given range to help us to draw plot to show the different areas by different color
ellipse = xx * 2 + yy**2
plt.contourf(x, y, ellipse, cmap = 'jet') # we assign one function to draw plot
plt.colorbar()  # show the color bar
plt.show() 


# Example 2
random_data = np.random.random((8, 6)) 
plt.contourf(x, y, random_data, cmap = 'inferno')   # we generate random data and plot it
plt.colorbar() 
plt.show() 


# Example 3
x = np.arange(-5, 5, 0.1)  # range from -5 to 5 with step 0.1
y = np.arange(-5, 5, 0.2)
# here we have 100 x and 50 y in 1D array
xx, yy = np.meshgrid(x, y)
# after meshgrid them, we get xx 
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z, cmap = 'jet')
plt.colorbar() 
plt.show()

# this code is modified from https://www.geeksforgeeks.org/numpy-meshgrid-function/
