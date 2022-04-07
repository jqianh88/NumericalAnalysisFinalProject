"""
numpyWorksheet.py
Name: Eric A. Autry
Date: 02/17/22
"""

import numpy as np # Common to give it a shorter nickname.

v = np.array([5,3,-2]) # 1D array (like all row or column vectors in numpy).

A = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 2D array for matrices.


# Define the rest of the matrices/vectors from the worksheet.
w = np.array([1, 5, -1]) # 1D array
B = np.array([[3, 1, 1], [2, 2, 4], [5,7,1]])

print()
print('slicing')
print('A\n',A,'\n') # np.array repr function prints matrices in a nice format.
print('A[0]\n',A[0],'\n')   # Gets the 0 row (since it is an array of arrays).
print('A[0,:]\n',A[0,:],'\n') # Gets the 0 row, and all columns (: means all).
print('A[:,0]\n',A[:,0],'\n') # Gets all rows of the 0 column.

# Try some slicing.
print('try some slicing')
print('A[-1]\n', A[-1], '\n')      # gets the last row?
print('A[1,:]\n', A[-1, :], '\n')  # gets the last row and all columns (: =
# all)
print('A[:,2]\n', A[:,2], '\n')  # get all rows and last column
print('A[(0,2),(1,2)]\n', A[(0,2),(1,2)], '\n') # get 2nd row and last 2
print('A[0:2,0:2]\n', A[0:2,0:2], '\n')        # 1st 2rows and 2 cols
print('A[[0,2],[1,2]]\n', A[[0,2],[1,2]], '\n') #gets first row column 2nd
# column entry and 3rd row and 3rd column entry and makes an array


print()
arr =np.arange(25) # gives 1D vector of 0-24
print('aranged:',arr,'\n')
fullMat = np.arange(25).reshape((5,5)) # arange gives 0-24, reshaped to 5x5.
print('reshaped:',fullMat,'\n')
mat = np.array([1,2,3,4,7,4,3,10,-2,7,9,-1]).reshape(3,4) # more reshape
print(mat)
mat = mat.reshape(12,1) # and some more reshape - note, an actual col vector!
print(mat)
print()

print('trying reshape, shape')
C = np.array([1,2,3,4,5,6])
print(C.shape)              # gets shape of the array/matrix
print(C.reshape(6,1))       #Trick to be column vector
D = np.zeros(C.shape)       # allocates a zero array of size C
print(D)


# Try to extract the inner 3x3 matrix from fullMat with slicing.
print('Challenge')
innerSquare = fullMat[0:3, 0:3]
print('full matrix:\n',fullMat,'\n')
print('inner square:\n',innerSquare,'\n')
print()

print('arithmetic')
# Try some arithmetic and print the results.
# Note: +=, -=, etc all work for matrices.
innerSquare += A
print(innerSquare)

print()
print('matrix multiplication with @')
B = np.arange(9).reshape((3,3)) # arange gives 0-8, reshaped to 3x3.
C = A@B # Matrix multiplication with @ in numpy.
print(C)
print()

# Compute the dot product of the first row of A and the 2nd column of B.
# FIXME

# Try multiplication using A, v, and w and see what happens.
# Note: what if we want the outer product of v and w?
# FIXME

print()
print('logical indexing')
print(A)
print(A>5) # All entries >5 flagged as True.
print(A[A>5]) # Gets all of those flagged entries.
print()

# Can you write code to extract all even elements of A?
# FIXME

print()
print('aggregation functions')
fullMat = np.arange(25).reshape((5,5)) # arange gives 0-24, reshaped to 5x5.
print(fullMat)
print('col mins:',fullMat.min(axis=0)) # take the min of each col (axis=0)
print('row mins:',fullMat.min(axis=1)) # take the min of each row (axis=1)
print()

print('zeros and ones')
mat0 = np.zeros((3,2)) # 3x2 Matrix of all 0s.
print(mat0)
mat1 = np.ones((2,3)) # 2x3 Matrix of all 1s.
print(mat1)
print()

print('transposes')
print(A)
print(A.T) # Take the transpose.
print()

# Try taking the transpose of vectors v and w. What happens?
# FIXME

print()
print('copies')
Acopy = A.copy() # copy method works for numpy arrays
print(A)
print(Acopy)
print()

print('ints vs floats')

A = np.arange(25).reshape((5,5)) # Create a 5x5 matrix.
print(A)
# A /= 2 # Uh oh! Can't do this! (try it and see what happens)
print()

A = np.arange(25,dtype='float').reshape((5,5)) # Matrix of floats!
print(A)
A /= 2 # Now it works!
print(A)
print()

A = np.array([[1.,2,3],[4,5,6],[7,8,9]]) # What does this do?
print(A) # Note that the single decimal on the 1 was enough to make all floats.
A /= 2 # Still works!
print(A)
print()

A = np.array([[1+0j,2,3],[4,5,6],[7,8,9]]) # What does this do?
print(A) # Complex entries! Note: j is the complex constant.
         # Also note: all values are floats since it is complex.
print()

# Can you write code to extract all non-zero columns from a matrix?
# FIXME

# Can you write code to insert a new column at the end of a matrix?
# FIXME




'''
ints and floats critically important for hw 4. The type is crucial. 
'''