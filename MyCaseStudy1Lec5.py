import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math as m

# Functions
def f1(x):
    y = x**2-x-6
    return y

def f2(x):
    y = x**3 - x**2
    return y

def f3(x):
    y = x**3 - 2*x**2 +(4/3)*x -8/27
    return y

def f4(x):
    y = x - 2**(-x)
    return y

# Derivatives
def df1(x):
    y = 2*x-1
    return y

def df2(x):
    y = 3*x**2 - 2*x
    return y

def df3(x):
    y = 3*x**2 - 4*x + 4/3
    return y

def df4(x):
    y = 1 + np.log(2)*2**(-x)
    return y

# Bisection method\n",
def bisectionmethodIntervalSelectionf1(a, b):        # To choose a length one interval.
    while (b - a)/2 > 0.5:
        middle = (a+b)/2
        if f1(a) == 0:
            return a, b
        elif f1(middle)*f1(a) < 0:
            b = middle
        else:
            a = middle
    return a, b



def bisectionmethodf1():
    a = 2  # initial interval for root
    b = 5  # initial interval for root
    N = 10  # number of iterations
    i = 0
    lamb = np.zeros(N)  # estimates of lambda
    alpha = np.zeros(N)  # estimates of order alpha

    # initalize |pn-p|-type differences
    diffA = 2
    diffB = 2
    diffC = 2
    p = 2

    while i < N:
        pNew = a + (b - a) / 2
        diffC = diffB
        diffB = diffA
        diffA = abs(pNew - p)
        p = pNew
        if f1(p) == 0:
            break
        if f1(a) * f1(p) > 0:
            a = p
        else:
            b = p
        alpha[i] = np.log(diffA / diffB) / np.log(diffB / diffC)
        lamb[i] = diffA / (diffB ** alpha[i])
        print(i, p, lamb[i], alpha[i])
        i = i + 1


'''

# Fixed Point Iteration method\n",
    p     = 2          # initial guess for root\n",
    c     = 1/3        # scaling factor\n",
    N     = 10         # number of iterations\n",
    lamb  = np.zeros(N) # estimates of lambda\n",
    alpha = np.zeros(N) # estimates of order alpha\n",
    i     = 0
    # initialize |pn-p|-type differences\n",
    diffA = 2
    diffB = 2
    diffC = 2

def gp(p):
    y = p-c*f1(p)
    return y

    while i<N:
        pNew  = gp(p)
        diffC = diffB
        diffB = diffA
        diffA = abs(pNew-p)
        p     = pNew
        alpha[i] = np.log(diffA/diffB)/np.log(diffB/diffC)
        lamb[i]  = diffA/(diffB**alpha[i])
        print(i,p,lamb[i],alpha[i])
        i     = i+1
        
        
# Newton's method\n",
    p     = 0          # initial guess for root\n",
    N     = 10         # number of iterations\n",
    lamb  = np.zeros(N) # estimates of lambda\n",
    alpha = np.zeros(N) # estimates of order alpha\n",
    i     = 0

    # initalize |pn-p|-type differences\n",
    diffA = 2
    diffB = 2
    diffC = 2

    while i<N:
        pNew  = p-f1(p)/df1(p) # Newton's method\n",
        diffC = diffB
        diffB = diffA
        diffA = abs(pNew-p)
        p     = pNew # update root estimate\n",
        alpha[i] = np.log(diffA/diffB)/np.log(diffB/diffC)
        lamb[i]  = diffA/(diffB**alpha[i])
        print(i,p,lamb[i],alpha[i])
        i     = i+1

'''
# Secant method\n",
def secantmethod():
    p     = 2   # initial guess for root\n",
    pOld  = 2.5 # initial guess for root\n",
    N     = 10  # number of iterations\n",
    lamb  = np.zeros(N) # estimates of lambda\n",
    alpha = np.zeros(N) # estimates of order alpha\n",
    i     = 0
    # initalize |pn-p|-type differences\n",
    diffA = 2
    diffB = 2
    diffC = 2
    while i<N:
        q    = f1(p)
        qOld = f1(pOld)
        pNew = p-q*(p-pOld)/(q-qOld) # Secant method\n",
        diffC = diffB
        diffB = diffA
        diffA = abs(pNew-p)
        pOld  = p
        p     = pNew # update root estimate\n",
        alpha[i] = np.log(diffA/diffB)/np.log(diffB/diffC)
        lamb[i]  = diffA/(diffB**alpha[i])
        print(i,p,lamb[i],alpha[i])
        i     = i+1









if __name__ == "__main__":

    func1 = f1(-2)

    func2 = f2(2)

    func3 = f3(2)

    func4 = f4(2)

    dfunc1 = df1(2)

    dfunc2 = df2(2)

    dfunc3 = df3(2)

    dfunc4 = df4(2)

    bm1a, bm1b = bisectionmethodIntervalSelectionf1()
    print(f'Function 1 Interval Selection is [{bm1a},{bm1b}].')
    bms1 = bisectionmethodf1()

    # Bisection Method Sample plot\n",
    from matplotlib import rc
    import matplotlib as mpl

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    import seaborn

    seaborn.set(style='ticks')

    fig = plt.figure()
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)

    # Create an axes instance\n",
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.scatter(range(10), bms1, s=100, linewidths=2.5, facecolors='none', edgecolors='b')
    ax.axis('equal')
    plt.xlabel('Number of iterations', size=16)
    plt.ylabel('Estimate of alpha', size=16)
    # plt.savefig('test.png',transparent=True)
    plt.show()
