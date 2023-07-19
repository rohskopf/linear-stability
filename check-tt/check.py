import numpy as np

amat_tt = np.loadtxt("../transpose-trick/amat_tt.dat")
amat_full = np.loadtxt("../full-matrix/amat_full.dat")

# TODO: Fix TT A which has 6 extra rows per config for stresses
print(np.shape(amat_tt))
print(np.shape(amat_full))

diff = amat_tt - amat_full
print(np.max(diff))