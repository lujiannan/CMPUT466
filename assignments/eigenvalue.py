import numpy as np
from numpy import linalg as LA
H = np.array([ [11, 12], [21, 22]])  # your values here
eigenval, eigenvec = LA.eig(H)