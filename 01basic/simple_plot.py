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
#plt.show()

# there is a convenince method plt.subplots, that creates a new figure and returns a Numpy array containing the created subplot objects

fig, axes = plt.subplots(2,2)
print axes
for i in range(2):
    for j in range(2):
        axes[i,j].scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
plt.subplots_adjust(wspace=0,hspace=0)
#plt.show()


# other details 

fig, ax = plt.subplots()
x = np.arange(0,2*np.pi,0.1)
y = np.sin(x)
# set the X axis ploting range
plt.xlim(0,10)
# set X axis ticks
ticks = ax.set_xticks([0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi])
# set lables for each ticks
labels = ax.set_xticklabels(['zero','one','two','three','four'],rotation=30,fontsize='small')
# set title
ax.set_title('the test plot')
# set X axis label
ax.set_xlabel('time')
# set legend
ax.plot(x,y,'k--',label='one')
ax.plot(x,np.abs(y),'k.',label='two')
ax.legend(loc='best')
plt.show()
