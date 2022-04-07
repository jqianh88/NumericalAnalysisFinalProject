
import math
import numpy
import sys




#2.1 #1 Gaussian elimination
def systemsofEquations():
    f1 = [[2,-2,-1], [4,1,-2], [-2,1,-1]]
    b1 = [-2, 1, -3]
    f2 = [[1, 2, -1], [0, 3, 1], [2, -1, 1]]
    b2 = [2, 4, 2]
    f3 = [[2, 1, -4], [1, -1, 1], [-1, 3, -2]]
    b3 = [-7, -2, 6]
    return [(f1, b1), (f2, b2), (f3, b3)]                   #list of tuples

def gaussianElimination(syst1, b1):
    m = len(b1)
    x = [0.0]*m

    # Forward elimination - column major
    for col in range(m):  # for each row loop over row, m = number of rows
        if abs(syst1[col][col]) < numpy.finfo(float).eps:               # less than epsilon
            raise ValueError('zero pivot encountered')
        for row in range(col + 1, m):  # loop over each column, n = number of cols
            ratio = syst1[row][col] / syst1[col][col]
            for c in range(col, m):
                syst1[row][c] = syst1[row][c] - ratio * syst1[col][c]
            b1[row] += -ratio * b1[col]

    # Back substitution - column major
    x[m - 1] = b1[m - 1] / syst1[m - 1][m - 1]
    for row in range(m - 1, -1, -1):
        for col in range(row + 1, m):
            b1[row] = b1[row] - (syst1[row][col] * x[col])
        x[row] = b1[row] / syst1[row][row]
    return x


# 2.1.2 Program problem Hilbert matrix
def hilbertMatrix(n):
    # list comprehension--space allocation for hmatrix b, and x
    hmatrix = [[1.0]*n for x in range(n)]
    for row in range(n):   # select row
        for column in range(n):   # loop across row so loop across columns [index]
            hmatrix[row][column] = 1/((row+1) + (column+1) - 1 )        # because python, add 1 to start from 1
    detHM2 = numpy.linalg.det(hmatrix)      #should be 1/12=.08333
    return hmatrix, detHM2

'''
# system 2
def gaussianElimination2():
    syst2 = systemsofEquations()[2]
    b2 = systemsofEquations()[3]
    m = len(b2)
    x = [0.0]*m

    # Forward Elimination - Row major
    for row in range(m-1):  # for each row loop over row, m = number of rows
        if abs(syst2[row][row]) == 0.0:
            raise ValueError('zero pivot encountered')
        for col in range(row+1, m):  # loop over each column, n = number of cols
            ratio = syst2[col][row]/syst2[row][row]
            #print('ratio', ratio)
            for c in range(row, m):
                syst2[col][c] = syst2[col][c] - ratio*syst2[row][c]
            b2[col] += -ratio * b2[row]
            #print('what', syst2[row][c], 'b2', b2[row])
    print('syst2, b2', syst2, b2)

     #Back substitution - row major
    #print(x[m - 1])
    #print(b2[m - 1], syst2[m - 1])
    x[m - 1] = b2[m - 1] / syst2[m - 1][m - 1]
    #print('this is x', x)
    for col in range(m-1, -1, -1):
        #print('row', row)
        for row in range(col+1, m):
            #print('row', row)
            b2[col] = b2[col] - (syst2[col][row] * x[row])
            #print('b2col', b2[col])
        x[col] = b2[col] / syst2[col][col]
        #print('x2col', x[col])
    return syst2, b2, x

#system 3
def gaussianElimination3():
    syst3 = systemsofEquations()[4]
    b3 = systemsofEquations()[5]
    m = len(b3)
    x = [0.0]*m

    # Forward Elimination - Row major
    for row in range(m-1):  # for each row loop over row, m = number of rows
        if abs(syst3[row][row]) == 0.0:
            raise ValueError('zero pivot encountered')
        for col in range(row+1, m):  # loop over each column, n = number of cols
            ratio = syst3[col][row]/syst3[row][row]
            #print('ratio', ratio)
            for c in range(row, m):
                syst3[col][c] = syst3[col][c] - ratio*syst3[row][c]
            b3[col] += -ratio * b3[row]
            #print('what', syst3[row][c], 'b3', b3[row])
    print('syst3, b3', syst3, b3)

     #Back substitution - row major
    #print(x[m - 1])
    #print(b3[m - 1], syst3[m - 1])
    x[m - 1] = b3[m - 1] / syst3[m - 1][m - 1]
    #print('this is x', x)
    for col in range(m-1, -1, -1):
        #print('row', row)
        for row in range(col+1, m):
            #print('row', row)
            b3[col] = b3[col] - (syst3[col][row] * x[row])
            #print('b3col', b3[col])
        x[col] = b3[col] / syst3[col][col]
        #print('xcol', x[col])
    return syst3, b3, x

'''









'''
def GEhilbert1():
    syst1 = hilbertMatrix2(2)[0]
    print(syst1)
    b1 = hilbertMatrix2(2)[1]
    detHM2 = hilbertMatrix2(2)[2]
    m = len(b1)
    x = [0.0]*m
    # Forward Elimination - Row major
    for row in range(m-1):  # for each row loop over row, m = number of rows
        if abs(syst1[row][row]) == 0.0:
            raise ValueError('zero pivot encountered')
        for col in range(row+1, m):  # loop over each column, n = number of cols
            ratio = syst1[col][row]/syst1[row][row]
            #print('ratio', ratio)
            for c in range(row, m):
                syst1[col][c] = syst1[col][c] - ratio*syst1[row][c]
            b1[col] += -ratio * b1[row]
            #print('what', syst1[row][c], 'b1', b1[row])
    print(f'2x2 Hilbert Matrix {syst1}, and b is {b1}.')

     #Back substitution - row major
    #print(x[m - 1])
    #print(b1[m - 1], syst1[m - 1])
    #x[m - 1] = b1[m - 1] / syst1[m - 1][m - 1]
    #print('this is x', x[m-1])
    for col in range(m-1, -1, -1):                              # problem with indexing, fix b1, mess up x, vice versa
        for row in range(col+1, m):
            #print('b1colorig', b1[col])
            #print('xrow', x[row])
            b1[col] += - (syst1[col][row] * x[row])           # - (syst1[col][row] * x[row])     #bug need to fix
            #print('b1col', b1[col])
        x[col] = b1[col] / syst1[col][col]
        #print('xcol', x[col])
    return syst1, b1, x, detHM2
    
def GEhilbert2():
    syst1 = hilbertMatrix5(5)[0]
    print(syst1)
    b1 = hilbertMatrix5(5)[1]
    detHM5 = hilbertMatrix5(5)[2]
    m = len(b1)
    x = [0.0]*m
    # Forward Elimination - Row major
    for row in range(m-1):  # for each row loop over row, m = number of rows
        if abs(syst1[row][row]) == 0.0:
            raise ValueError('zero pivot encountered')
        for col in range(row+1, m):  # loop over each column, n = number of cols
            ratio = syst1[col][row]/syst1[row][row]
            #print('ratio', ratio)
            for c in range(row, m):
                syst1[col][c] = syst1[col][c] - ratio*syst1[row][c]
            b1[col] += -ratio * b1[row]
            #print('what', syst1[row][c], 'b1', b1[row])
    print(f'5x5 Hilbert Matrix after GE is {syst1}, and b is {b1}.')

     #Back substitution - row major
    #print(x[m - 1])
    #print(b1[m - 1], syst1[m - 1])
    #x[m - 1] = b1[m - 1] / syst1[m - 1][m - 1]
    #print('this is x', x[m-1])
    for col in range(m-1, -1, -1):                              # problem with indexing, fix b1, mess up x, vice versa
        for row in range(col+1, m):
            #print('b1colorig', b1[col])
            #print('xrow', x[row])
            b1[col] += - (syst1[col][row] * x[row])           # - (syst1[col][row] * x[row])     #bug need to fix
            #print('b1col', b1[col])
        x[col] = b1[col] / syst1[col][col]
        #print('xcol', x[col])
    return syst1, b1, x, detHM5

def GEhilbert3():
    syst1 = hilbertMatrix10(10)[0]
    #print(syst1)
    b1 = hilbertMatrix10(10)[1]
    detHM10 = hilbertMatrix10(10)[2]
    m = len(b1)
    x = [0.0]*m
    # Forward Elimination - Row major
    for row in range(m-1):  # for each row loop over row, m = number of rows
        if abs(syst1[row][row]) == 0.0:
            raise ValueError('zero pivot encountered')
        for col in range(row+1, m):  # loop over each column, n = number of cols
            ratio = syst1[col][row]/syst1[row][row]
            #print('ratio', ratio)
            for c in range(row, m):
                syst1[col][c] = syst1[col][c] - ratio*syst1[row][c]
            b1[col] += -ratio * b1[row]
            #print('what', syst1[row][c], 'b1', b1[row])
    print(f'5x5 Hilbert Matrix after GE is {syst1}, and b is {b1}.')

     #Back substitution - row major
    #print(x[m - 1])
    #print(b1[m - 1], syst1[m - 1])
    #x[m - 1] = b1[m - 1] / syst1[m - 1][m - 1]
    #print('this is x', x[m-1])
    for col in range(m-1, -1, -1):                              # problem with indexing, fix b1, mess up x, vice versa
        for row in range(col+1, m):
           # print('b1colorig', b1[col])
            #print('xrow', x[row])
            b1[col] += - (syst1[col][row] * x[row])           # - (syst1[col][row] * x[row])     #bug need to fix
           # print('b1col', b1[col])
        x[col] = b1[col] / syst1[col][col]
        #print('xcol', x[col])
    return syst1, b1, x, detHM10


def hilbertMatrix5(n):
    # list comprehension--space allocation
    hmatrix = [[1.0, 1.0, 1.0, 1.0, 1.0] for x in range(n)]
    # space allocation for b and x
    b2 = [1.0] * n
    for row in range(n):   # select row
        for column in range(n):   # loop across row so loop across columns [index]
            hmatrix[row][column] = 1/((row+1) + (column+1) - 1 )       # because python, add 1 to start from 1
    detHM5 = numpy.linalg.det(hmatrix)
    #print('determinant', detHM5)
    return hmatrix, b2, detHM5



def hilbertMatrix10(n):
    # list comprehension--space allocation
    hmatrix = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] for x in range(n)]
    # space allocation for b and x
    b3 = [1.0] * n
    for row in range(n):   # select row
        for column in range(n):   # loop across row so loop across columns [index]
            hmatrix[row][column] = 1/((row+1) + (column+1) - 1 )       # because python, add 1 to start from 1
    detHM10 = numpy.linalg.det(hmatrix)
    #print('determinant', detHM10)
    return hmatrix, b3, detHM10

'''





'''
  # Forward elimination - column major 
    for col in range(m):  # for each row loop over row, m = number of rows
        if abs(syst1[col][col]) == 0.0:
            raise ValueError('zero pivot encountered')
        for row in range(col + 1, m):  # loop over each column, n = number of cols
            ratio = syst1[row][col] / syst1[col][col]
            print('ratio', ratio)
            for c in range(col, m):
                syst1[row][c] = syst1[row][c] - ratio * syst1[col][c]
            b1[row] += -ratio * b1[col]
            print('what', syst1[col][c], 'b1', b1[col])
    print('syst1, b1', syst1, b1)

    #Back substitution - column major
    print(x[m - 1])
    print(b1[m - 1], syst1[m - 1])
    x[m - 1] = b1[m - 1] / syst1[m - 1][m - 1]
    print('this is x', x)
    for row in range(m-1, -1, -1):
        print('row', row)
        for col in range(row+1, m):
            print('col', col)
            b1[row] = b1[row] - (syst1[row][col] * x[col])
            print('b1row', b1[row])
        x[row] = b1[row] / syst1[row][row]
        print('xrow', x[row])
    return syst1, b1, x
    
    
    
    
    
    
    
    # Forward Elimination - Row major
    for row in range(m-1):  # for each row loop over row, m = number of rows
        if abs(syst1[row][row]) < numpy.finfo(float).eps:               # less than epsilon
            raise ValueError('zero pivot encountered')
        for col in range(row+1, m):  # loop over each column, n = number of cols
            ratio = syst1[col][row]/syst1[row][row]
            for c in range(row, m):
                syst1[col][c] = syst1[col][c] - ratio*syst1[row][c]
            b1[col] += -ratio * b1[row]
    print('syst1, b1', syst1, b1)

     #Back substitution - row major
    x[m - 1] = b1[m - 1] / syst1[m - 1][m - 1]
    for col in range(m-1, -1, -1):
        for row in range(col+1, m):
            b1[col] = b1[col] - (syst1[col][row] * x[row])
        x[col] = b1[col] / syst1[col][col]
    return x
'''




'''
    https://www.overleaf.com/8574216924krjyxjzvdcvx
'''

 #syst2, b2, syst3, b3


if __name__ == "__main__":
    list_systeq = systemsofEquations()
    for systeq in list_systeq:
        systA = systeq[0]
        systb = systeq[1]
        x1 = gaussianElimination(systA, systb)
        print(f'Result of GE is {systA}, b1 is {systb}, and the solution x is {x1}.')
    print(' ')

    n_list = [2, 5, 10]
    for n in n_list:
        hb = [1.0]*n
        HM, det = hilbertMatrix(n)
        print('HM', HM, hb, 'Determinant', det)
        hilbert = gaussianElimination(HM, hb)
        print('result of hilbert', hilbert)

### if you're copying then it's a loop
    '''
    hilbert2 = hilbertMatrix2(2)
    print(hilbert2)
    print(' ')
    hilbert5 = hilbertMatrix5(5)
    print(f'This is {hilbert5}.')
    print(' ')
    hilbert10 = hilbertMatrix10(10)
    print(hilbert10)
    print(' ')
    hilbert2x2, b, x, det = GEhilbert1()
    print(f'The resulting 2x2 Hilbert matrix is {hilbert2x2}, b1 is {b}, the solution x is {x}, and the '
          f'determinant is {det}.')
    print(' ')
    hilbert5x5, b, x, det = GEhilbert2()
    print(f'The resulting 5x5 Hilbert matrix is {hilbert5x5}, b2 is {b}, the solution x is {x}, and the '
          f'determinant is {det}.')
    print(' ')
    hilbert10x10, b, x, det = GEhilbert3()
    print(f'The resulting 10x10 Hilbert matrix is {hilbert10x10}, b3 is {b},  the solution x is {x}, and the '
          f'determinant is {det}.')
    
    '''