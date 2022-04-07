


import numpy as np
import matplotlib.pyplot as plt


#Program 6.1 Euler’s Method for Solving Initial Value Problems
#Use with ydot.m to evaluate rhs of differential equation
# Input: interval inter, initial value y0, number of steps n
# Output: time steps t, solution y
# Example usage: euler([0 1],1,10)
# function [t,y]=euler(inter,y0,n) t(1)=inter(1)
# y(1)=y0
# h=(inter(2)-inter(1))/n
# n = (b-a)/h, h = step size
#a,b = interval
# y0 = initial value = y(a)

def euler(f, a, b, y0, h):
    n = int((b-a)/h)
    t = np.zeros(n+1, dtype='float')
    y = np.zeros(n+1)
    print(y)
    t[0] = a
    y[0] = y0
    for i in range(1,n+1):
        print(i)
        t[i] = t[i-1] + h
        y[i] = y[i-1] + h*f(t[i-1], y[i-1])
        print(t[i])
        print(y[i])
    return t,y

'''
Adaptive Quardrature Trapezoid method 
'''
# Computes approximation to definite integral
# Inputs: Matlab function f, interval [a0,b0],
# error tolerance tol0
# Output: approximate definite integral
def ad_trap_quad(f, a, b, tol):
    return _rec_ad_trap_quad(f, a, b, a, b, tol)

def _rec_ad_trap_quad(f, a, b, a0, b0, tol):
    c = (a+b)/2
    Sab = _trap_area(f, a, b)
    Sac = _trap_area(f, a, c)
    Scb = _trap_area(f, c, b)
    if np.abs(Sab-Sac-Scb) < 3* tol * (b-a)/(b0-a0):
        return Sac + Scb, 2
    else:
        sum1, n1 = _rec_ad_trap_quad(f, a, c, a0, b0, tol)
        sum2, n2 = _rec_ad_trap_quad(f, c, b, a0, b0, tol)
        return sum1+sum2, n1 + n2

def _trap_area(f, a, b):
    return (b-a)*(f(a)+f(b))/2

'''
For Adaptive Quadrature Trapezoid Method 
'''
def faAQD(x):
    return 2*np.sqrt(1-x**2)

def fbAQD(x):
    return 1 + np.sin(np.exp(3*x))

'''
For the other functions 
'''

def fa(t,y):
    return t

def fb(t,y):
    return (t**2) * y

def fc(t,y):
    return 2*(t+1)*y

def fa3(t,y):
    return t+y

def fb3(t,y):
    return t-y

def fc3(t, y):
    return (4*t)-(2*y)


def faexact(t):
    return ((t**2)/2) + 1

def fbexact(t):
    return np.exp((t**3)/3)

def fcexact(t):
    return np.exp((t**2)+(2*t))

def faexact3(t):
    return -t-1+np.exp(t)

def fbexact3(t):
    return t-1+np.exp(-t)

def fcexact3(t):
    return 2*t -1 + (1/np.exp(2*t))


### To do:
# Change z, interval, and number of steps.

if __name__ == "__main__":
    hList = [0.1, 0.05, 0.025]
    x = np.linspace(0, 1, 30)               # 30 linearly spaced numbers
    functionName = ["t", "t^2y", "2(t+1)y"]
    functionName3 = ["t+y", "t-y", "4t-2y"]
    functionList = [fa, fb, fc]
    exactSol = [faexact(x), fbexact(x), fcexact(x)]
    functionList3 = [fa3, fb3, fc3]
    exactSol3 = [faexact3(x), fbexact3(x), fcexact3(x)]
    qname = ['a', 'b', 'c']
    a = 0
    b = 1
    y0Q2 = 1
    y0Q3 = 0


    figurenum = 1

    for function, name, exact, question in zip(functionList, functionName,
                                           exactSol,
                                     qname):
        plt.figure(figurenum)
        for step in hList:
            t,y = euler(function, a, b, y0Q2, step)
            plt.plot(t,y, label = f'Function: dy/dx={name}, Step size: {step}')
        plt.plot(x, exact, label = f"Exact Solution of dy/dx = {name}")
        plt.xlabel('Time Steps h')
        plt.ylabel('Solutions')
        plt.title(f'Estimated and Exact Solutions to Q2 Function {question}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'HW7Q2{question}.png', bbox_inches='tight')
        figurenum +=1


    for function, name3, exact3, question in zip(functionList3, functionName3,
                                       exactSol3, qname):
        plt.figure(figurenum)
        for step in hList:
            t,y = euler(function, a, b, y0Q3, step)
            plt.plot(t,y, label = f'Function: dy/dx={name3}, Step size: {step}')
        plt.plot(x, exact3, label = f"Exact Solution of dy/dx = {name3}")
        plt.xlabel('Time Steps h')
        plt.ylabel('Solutions')
        plt.title(f'Estimated and Exact Solutions to Q3 Function {question}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'HW7Q3{question}.png', bbox_inches='tight')
        figurenum +=1


    '''    
    plt.figure(2)
    for step in hList:
        for function, y0 in zip(functionList, ivalue):
            t, y = euler(function, a, b, y0, step)
            plt.plot(t, y, label=f'{function}')
        plt.xlabel('h')
        plt.ylabel('Error')
        plt.title('Loglog Plot h vs error')
        plt.legend(loc='best')
        plt.grid(True)
    '''
    #plt.show()

    '''
    Numerical Integration values for Adaptive Quadrature Method
    '''
    f_list = [faAQD, fbAQD]
    aAQD = -1
    bAQD = 1
    tol = 0.5 * 10 ** (-8)  # first TOL
    print('Area               N')
    print('========         ========')
    for i, func in enumerate(f_list):
        area, N = ad_trap_quad(func, aAQD, bAQD, tol)
        print(f'{area:.8f}      {N}')








