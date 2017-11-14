#!/usr/bin/python

# this is a simple program to predict the GMV of 11.11 of alibaba

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# history data
year = range(2009,2017)
gmv = [0.52,9.36,52,191,362,571,912,1207]
fig, ax = plt.subplots()
plot1 = ax.plot(year,gmv,'*',label='original values')

# fit curve
res = np.polyfit(year,gmv,3)
year = range(2009,2018)
model = np.poly1d(res)
pred = model(year)
print pred
plot2 = ax.plot(year,pred,'r',label='predict values')
plt.xticks(year,year)
plt.xlabel('years')
plt.ylabel('GMV')
plt.title('ALIPAPA GMV')
plt.show()
#plt.savefig('p1.png')

