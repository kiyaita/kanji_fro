import numpy as np

Z= np.empty([])
A = [[11,12,13]]
Aq = np.array([[11,12,13],
              [21,22,23]])
"""B = np.array([2,2,2])
C=np.vstack((B,A))
for i in range(50):
    if np.ndim(Z) == 0: Z= A
    else:Z=np.vstack((Z,B))"""
print(np.ndim(A))
for a in A:
    print(int(a[0]), int(a[1]))