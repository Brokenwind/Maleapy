#!/usr/bin/python

import numpy as np
import pandas as pd

# Since this is comma-delimited, we can use read_csv to read it into DataFrame
#data = pd.read_csv('ex1data1.txt')

# we could  use read_table and specify the delimiter
data = pd.read_table('ex1data1.txt',sep=',')

# create a matrix with DataFrame
ma = np.mat(data)
print type(ma)

# the transpose of T multiply
ma.T * ma
print data
