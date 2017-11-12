#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
ax3.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
ax4.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
plt.show()

# there is a convenince method plt.subplots, that creates a new figure and returns a Numpy array containing the created subplot objects

fig, axes = plt.subplots(2,2)
print axes
for i in range(2):
    for j in range(2):
        axes[i,j].scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()


