import numpy as np
import matplotlib.pyplot as plt
'''Use the three-point centered-difference formula for the second derivative to
approximate f"(0), where f (x) = cos(x), as in problem 1. 
Find the approximation error.'''

'''Compute the difference formula for f'(a) with step size h.

Parameters
----------
f : function
    Vectorized function of one variable
p : number
    Compute derivative at x = p  # it will be 1 in this case
method : string
    Difference formula: 'forward', 'backward' or 'central'
h : float
    Step size in difference formula

Returns
-------
float
    Difference formula for 2nd derivative:
        central: (f(p+h) + f(p-h) - 2f(p)/(h**2)

'''

def derivative(f,p,h, method='central'):
    if method == 'central':
        return (f(p+h) - (2*f(p)) + f(p-h))/(h**2)





# enumerate(), looking for index and number
# zip() takes two lists and returns set of tuples of each element
if __name__ == "__main__":
    varlist = range(1,13)
    hlist = [10**(-var) for var in varlist]
    p = 0
    fb = lambda x: np.cos(x)
    errorlist = []
    correctans = -np.cos(0)
    for h in hlist:
        dydx = derivative(fb,p,h)
        error = abs(correctans - dydx)
        errorlist.append(error)
        plt.figure(1)
        plt.plot(0, dydx, 'o', label=f'Central Difference y=f"(x), for h = '
                                     f'{h}')
        plt.xlabel('x-coordinate')
        plt.ylabel('Approximated Derivative')
        plt.title('Approximation of the Derivative of cos(x) evaluated at 0')
        plt.legend(loc = 'best')
        plt.grid(True)
    print(dydx)
    print()
    print(errorlist)
    for x,y in zip(hlist,errorlist):
        plt.figure(2)
        plt.plot(np.log10(x), np.log10(y), 'o', label=f'Error for h ='
                                                   f' {x}')
        plt.xlabel('h')
        plt.ylabel('Error')
        plt.title('Loglog Plot h vs error')
        plt.legend(loc='best')
        plt.grid(True)
    #plt.show()
    exponent = (1/4)
    me = 2**-52
    minerr = 10**-4
    check = me**(exponent)
    print(check, minerr)
# minimum error appears to occur at 10**-4


