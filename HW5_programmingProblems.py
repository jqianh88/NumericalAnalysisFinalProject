import numpy as np
import matplotlib.pyplot as plt
from HW4_ProgrammingProblems import gaussianElimination

# Part 1
# Program 3.1 Newton Divided Difference Interpolation Method #Computes coefficients of interpolating polynomial
# Input: x and z are vectors containing the x and z coordinates # of the n data points
# Output: coefficients c of interpolating polynomial in nested form #Use with nest.m to evaluate interpolating polznomial
def newtdd(a,b,n):      # a = x, b = z
    v = np.zeros((n,n))        # initialize matrix
    c = np.zeros(n)     # passing tuple
    for j in range(n):
        v[j][0]=b[j]            # Fill in z column of Newton triangle
    for i in range(1,n):      # For column i,
        for j in range(n-i):    # fill in column from top to bottom
            v[j][i]=(v[j+1][i-1]-v[j][i-1])/(a[j+i]-a[j])
    for i in range(n):
        c[i]=v[0][i]        # Read along top of triangle
    return c


#Program 0.1 Nested multiplication
#Evaluates polynomial from nested form using Horner’s Method #Input: degree d of polynomial,
#       array of d+1 coefficients c (constant term first),
# x-coordinate x at which to evaluate, and
# array of d base points b, if needed #Output: value y of polynomial at x
### compute the coeff, compute polynomial,
def nest(c,a,x):            # c = coeff, a = output base coordinates, x= unknown input
    d = len(c)-1
    y = c[d]
    for i in range(d-1,-1,-1):
        y = y * (x-a[i])+c[i]
    return y



# Part 2: Natural Cubic Spline
# Program 3.5 Calculation of spline coefficients
# Calculates coefficients of cubic spline
# Input: x,y vectors of data points
# plus two optional extra data v1, vn
#Output: matrix of coefficients b1,c1,d1 b2,c2,d2 ... 
def splinecoeff(x,y,splinetype):
    n=len(x)
    v1=0
    vn=0
    dx = np.zeros((n-1,1))
    dy = np.zeros((n-1,1))
    A=np.zeros((n,n))        # matrix A is nxn 
    r=np.zeros((n,1))        # nx1
    for i in range(0,n-1):      # define the deltas
        dx[i]= x[i+1]-x[i]
        dy[i]=y[i+1]-y[i] 
    for i in range(1, n-1):   # load the A matrix
        A[i,i-1:i+2]=[dx[i-1], 2*(dx[i-1]+dx[i]), dx[i]]
        r[i]=3*(dy[i]/dx[i]-dy[i-1]/dx[i-1])
        
# Set endpoint conditions
# Use only one of following 5 pairs:
    if splinetype == 'natural':
        A[0][0] = 1 # natural spline conditions
        A[n-1][n-1] = 1
    #A(1,1)=2
    # r(1)=v1       # curvature-adj conditions
    #A(n,n)=2 r(n)=vn
    if splinetype == 'clamped':
        A[0][0:2]=[2*dx[0], dx[0]]
        r[0]=3*(dy[0]/dx[0]-v1)     #clamped
        A[n-1][n-2:n]=[dx[n-2], 2*dx[n-2]]
        r[n-1]=3*(vn-dy[n-2]/dx[n-2])
    #A(1,1:2)=[1 -1]        # parabol-term conditions, for n>=3
    #A(n,n-1:n)=[1 -1]
    #A(1,1:3)=[dx(2) -(dx(1)+dx(2)) dx(1)] # not-a-knot, for n>=4
    #A(n,n-2:n)=[dx(n-1) -(dx(n-2)+dx(n-1)) dx(n-2)]
    coeff=np.zeros((n,3))
    #print('This is A, r', A, r)
    coeff[:,1]= gaussianElimination(A, r)       # solve for c coefficients
    for i in range(n-1):          # solve for b and d
        coeff[i][2]=(coeff[i+1][1]-coeff[i][1])/(3*dx[i])
        coeff[i][0]=dy[i]/dx[i]-dx[i]*(2*coeff[i][1]+coeff[i+1][1])/3
    coeff=coeff[0:n-1,:]
    return coeff



# Program 3.6 Cubic spline plot
# Computes and plots spline from data points
# Input: x,y vectors of data points, number k of plotted points per segment
# Output: x1, y1 spline values at plotted points
def splineplot(x,y,k, splinetype):
    n=len(x) 
    coeff=splinecoeff(x,y, splinetype)
    x1=np.array([], dtype = np.float64)
    y1=np.array([], dtype = np.float64)
    for i in range(n-1):
        xs=np.linspace(x[i],x[i+1],k+1)
        dx=xs-x[i] 
        ys=coeff[i][2]*dx        #  evaluate using nested multiplication
        ys=(ys+coeff[i][1]) *dx
        ys=(ys+coeff[i][0]) *dx+y[i]
        x1 = np.append(x1, np.transpose(xs[0:k]))
        y1 = np.append(y1, np.transpose(ys[0:k]))
        print('x1', x1)
        print()
        print(y1)
    x1=np.append(x1,  x[n-1])
    y1=np.append(y1, y[n-1])
    print('x1,y1', x1, y1)
    return x1, y1





# Part 3: Clamped Cubic Spline
    '''

def splinecoeff(x, y):
    n = len(x)
    m = len(y)
    v1 = 0
    vn = 0
    dx = [0.0 for entry in range(n)]        # initialize list comp
    dy = [0.0 for entry in range(m)]        # initialize list comp
    A = np.zeros((n, n))  # matrix A is nxn
    r = np.zeros((n, 1))
    for i in range(0, n - 1):  # define the deltas
        dx[i] = x[i + 1] - x[i]
        dy[i] = y[i + 1] - y[i]
    for i in range(1, n - 1):  # load the A matrix
        A[i, i - 1:i + 1] = [dx[i - 1], 2 * (dx[i - 1] + dx[i]), dx[i]]  # how do I convert syntax
        r[i] = 3 * (dy[i] / dx[i] - dy[i - 1] / dx[i - 1])  # right-hand side, index check needed
        
    '''

'''

# make sure to define all arrays of zeros
# b and d and matrix defined
# focus on matrix that determines all values of c
#- start with that matrix
# make an array with all x and y
# build 2 more arrays, that have all upper and lowercase deltas (differences between x and ys
# build nxn matrix that follow this formula
# start off with matrix of zeros, manually put top left and bottom right = 1
# for loop putting entries into 3 entries per row
# runs 2nd row to 2nd to last row
# then comput inverse of matrix then multiply inverse matrix by vector
# build vector with for loop
# compute and then get c1 through cn
# b-coeff, defined in terms c and d coeff.
# find cs then bs and ds are explicit formulas
# sampling of inputs to plot
# use linspace command to test polynomials


#natural vs clamped
# clamped changes 1 in the top left and bottom right by calculation to change those
# values and their adjacent 0s may change and the first and last entry of the b vector
'''
if __name__ == "__main__":

    # Inputs
    des_cols = (0, 2)
    total_rows = 29
    file_name = "shots.txt"
    num_pts_list = [3,5,8]
    num_x_plot = 100

    # Original Data Points
    load_data = np.loadtxt(file_name, usecols=(0, 2))
    desired_lines = load_data[0:29]           # All 29 points
    xOD = desired_lines[:, 0].tolist()
    yOD = desired_lines[:, 1].tolist()

    # Final Version
    for num_pts in num_pts_list:
        deln = int((total_rows - 1) / (num_pts - 1))
        load_data_3 = np.loadtxt(file_name, usecols=(0, 2))
        desired_lines = load_data_3[0:total_rows:deln]
        x = desired_lines[:, 0].tolist()
        z = desired_lines[:, 1].tolist()
        c = newtdd(x, z, num_pts)
        inputs = np.linspace(min(x) - 1, max(x) + 1,
                             num_x_plot)  # array of numbers from 0 to 100 that are evenly spaced
        outputs = np.zeros(100)
        for ind in range(len(inputs)):
            outputs[ind] = nest(c, x, inputs[ind])
        plt.figure(1)
        plt.xlabel('x-coordinate')
        plt.ylabel('z-coordinate')
        plt.plot(inputs, outputs, label = f'{num_pts} points line')
    plt.plot(xOD, yOD, 'o', label = 'Original Data')
    plt.legend(loc = 'best')
    plt.title('Interpolation Polynomials for 3, 5, and 8 points with Original Data')
   #plt.show()


    # Interpolation Polynomial Call
    del8 = int(28/7)                    # specifically for 8 points
    load_data_8 = np.loadtxt(file_name, usecols=(0, 2))
    desired_lines8 = load_data_8[0:29:del8]           # specifically for 8 points
    xOD8 = desired_lines8[:, 0].tolist()
    yOD8 = desired_lines8[:, 1].tolist()
    c = newtdd(xOD8, yOD8, 8)
    inputs = np.linspace(min(xOD8) - 1, max(xOD8) + 1,
                         num_x_plot)  # array of numbers from 0 to 100 that are evenly spaced
    outputs = np.zeros(100)
    for ind in range(len(inputs)):
        outputs[ind] = nest(c, xOD8, inputs[ind])

    # Cubic Spline Calls
    k = 5
    x1n, y1n = splineplot(xOD8, yOD8, k, 'natural')       #natural 8 points
    x1c, y1c = splineplot(xOD8, yOD8, k, 'clamped')       # clamped 8 points
    x1nOD, y1nOD = splineplot(xOD, yOD, k, 'natural')  # natural All data
    x1cOD, y1cOD = splineplot(xOD, yOD, k, 'clamped')  # clamped all data

    plt.figure(2)
    plt.xlabel('x-coordinate')
    plt.ylabel('z-coordinate')
    plt.plot(x1n,y1n, label='8 Points Natural Cubic Spline')             # 8 natural spline
    plt.plot(x1c,y1c, label='8 Points Clamped Cubic Spline')             # 8 clamped
    plt.plot(x1nOD,y1nOD, label='All Points Natural Cubic Spline')             # All points natural spline
    plt.plot(x1cOD,y1cOD, label='All Points Clamped Cubic Spline')             # All Points clamped
    plt.legend(loc='best')
    plt.title('Cubic Spline Curves with 8 points and All Points')

    plt.figure(3)
    plt.xlabel('x-coordinate')
    plt.ylabel('z-coordinate')
    plt.plot(xOD,yOD,'o', label = 'Original Data')               # real data
    plt.plot(x1n,y1n, label='Natural Cubic Spline')             # 8 natural spline
    plt.plot(x1c,y1c, label='Clamped Cubic Spline')             # 8 clamped
    plt.plot(inputs, outputs, label='Interpolating Polynomial')
    plt.legend(loc='best')
    plt.title('All 8 Point Cubic Spline Curves and Original Data')
    plt.show()

    # generate the data, xsp, ysp...
    #




    # Load data manually
    '''
    load_data_3 = np.loadtxt(file_name, usecols=(0, 2))
    desired_lines = load_data_3[0:29:14]
    x = desired_lines[:,0].tolist()
    z = desired_lines[:,1].tolist()

    # Load data of 5 points
    load_data_5 = np.loadtxt(file_name, usecols=(0, 2))
    desired_lines5 = load_data_5[0:29:7]
    x5 = desired_lines5[:, 0].tolist()
    z5 = desired_lines5[:, 1].tolist()

    # Load data of 8 points
    load_data_8 = np.loadtxt(file_name, usecols=(0, 2))
    desired_lines8 = load_data_8[0:29:4]
    x8 = desired_lines8[:, 0].tolist()
    z8 = desired_lines8[:, 1].tolist()
    '''

    '''for num_pts in num_pts_list:
        deln = int((total_rows - 1) / (num_pts - 1))
        load_data_3 = np.loadtxt(file_name, usecols=(0, 2))
        desired_lines = load_data_3[0:total_rows:deln]
        x = desired_lines[:, 0].tolist()
        z = desired_lines[:, 1].tolist()
        c = newtdd(x, z, num_pts)
        inputs = np.linspace(min(x) - 1, max(x) + 1,
                             num_x_plot)  # array of numbers from 0 to 100 that are evenly spaced
        outputs = np.zeros(100)
        for ind in range(len(inputs)):
            outputs[ind] = nest(c, x, inputs[ind])
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.plot(inputs, outputs, label = f'{num_pts} points line')
    plt.legend(loc = 'best')
    plt.show()


# test

    print('test')
    xtest = [0, 2, 3, 4, 7]
    ytest = [1,8,4, 3, 8]
    ctest = newtdd(xtest, ytest, len(xtest))
    print('test3')
    xtest3 = [74.3119, 75.7512, 78.5256, 80.5347, 83.0387, 85.4484, 87.7498, 89.5493]
    ytest3 = [7.75603, 9.78756, 11.7546, 12.5344, 13.3183, 12.6839, 11.4394, 9.82346]
    ctest3 = newtdd(xtest3, ytest3, len(xtest3))
    print('check ctest')
    #print(ctest)
    inputs = np.linspace(min(xtest3)-1,max(xtest3)+1,num_x_plot)           # array of numbers from 0 to 4 that are evenly spaced
    outputs = np.zeros(100)
    for ind, other in enumerate(inputs):           #range(len(inputs)):
        outputs[ind] = nest(ctest3, xtest3, inputs[ind])
        print('check it', outputs[ind])

'''

'''for i, num_pts in enumerate(num_pts_list):
    deln = int((total_rows-1)/(num_pts-1))
    load_data_3 = np.loadtxt(file_name, usecols=(0, 2))
    data = load_data_3[0:29:deln]

    #data = np.loadtxt(file_name, skiprows=deln, usecols=des_cols)
    x_traj = data[:, 0]
    z_traj = data[:, 1]
    x_plot = np.linspace(min(x_traj), max(x_traj), num_x_plot)

    c = newtdd(x_traj, z_traj, deln)
    z_polz = np.zeros((len(x_plot), 1))
    for j, x in enumerate(x_plot):
        z_polz[j] = nest(c, x, x_traj)

    plt.figure(1)
    plt.plot(x_plot, z_polz)
plt.show()'''


    #
    # code up nest routine
    # generate test data from 3.4, 3.5  run through dd and nest program


    #print(load_data_3)







