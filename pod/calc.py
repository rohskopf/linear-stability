import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys

mat = np.loadtxt("pod.dat")
print(np.shape(mat))
cond = LA.cond(mat) #/1e9
print(cond)

inveps = 1./(sys.float_info.epsilon)
ratio = cond / inveps

print(ratio)

#plt.imshow(mat, interpolation='none')
#plt.colorbar()
#plt.show()