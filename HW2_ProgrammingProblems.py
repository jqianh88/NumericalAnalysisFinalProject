import math
"""
This is a function to use bisection method. 
Its inputs are a and b which is the selected interval [a, b]. 
It outputs the number of iterations it takes to be within 10^-10 accuracy of 0. 
"""

def bisectionmethod(a, b):
    count = 0
    c = 1
    f = c**4 - c**3 - 10
    g = a**4 - a**3 - 10
    while (b-a)/2 > 10**-10:
        count += 1
        c = (a + b)/2
        if f == 0:
            break
        elif g*f < 0:
            print(g*f)
            b = c
            print(b, c)
        else:
            a = c
    print(count)


"""

1.1 Computer problem #1 (a, c)
Use the Bisection method to find the root to six correct decimal places. (a) x^3 = 9
c) cos^29x) +6 =x

"""
def intervalSelectioncompa(a_0, b_0):        # To choose a length one interval.
    f = a_0 ** 3 - 9
    g = b_0 ** 3 - 9
    while (b_0 - a_0) != 1 and f*g >= 0:
        middle = (b_0-a_0)/2
        if f == 0:
            return a_0, b_0
        elif g*f < 0:
            b_0 = middle
        else:
            a_0 = middle
    return a_0, b_0

def bisectionmethod_compa(a_0, b_0):          # To approximate the root.
    interval = intervalSelectioncompa(a_0, b_0)
    a = interval[0]
    b = interval[1]
    f = a**3 - 9
    g = b**3 - 9
    while (b-a)/2 > 0.5 * (10**-6):
        c = (a + b)/2
        f = c**3 - 9
        if f == 0:
            return c
        if g*f < 0:
            #print(g*f)
            a = c
        else:
            b = c
        c = round(c, 6)
    return c



def intervalSelectioncompb(a_0, b_0):       # To choose a length one interval.
    f = math.cos(a_0)**2 + 6 - a_0
    g = math.cos(b_0)**2 + 6 - b_0
    while (b_0 - a_0) != 1 and f * g >= 0:
        middle = (b_0 - a_0) / 2
        if f == 0:
            return a_0, b_0
        elif g * f < 0:
            b_0 = middle
        else:
            a_0 = middle
    return a_0, b_0


def bisectionmethod_compb(a, b):            # To approximate the root.
    f = math.cos(a)**2 + 6 - a
    g = math.cos(b)**2 + 6 - b
    while (b-a)/2 > 0.5 * (10**-6):
        c = (a + b)/2
        f = math.cos(c) ** 2 + 6 - c
        if f == 0:
            return c
        if g*f < 0:
            a = c
        else:
            b = c
        c = round(c, 6)
    return c


""" 
# 1.2 Computer problem #3 (b) 
Calculate the square roots of the following numbers to eight correct decimal places by using 
fixed point iteration as in example 1.6: for b) 5
State your initial guess and the number of steps needed. 
"""

# Initial guess x_0 = 2

def fpiMethod(x_0, target):
    count = 1
    x_1 = (x_0 + (target/x_0))/2.0
    while abs(x_1 - x_0) > 0.5 * (10**-8):
        x_0 = x_1
        x_1 = (x_0 + (target/x_0))/2.0
        count += 1
    return round(x_1, 8), count






if __name__ == "__main__":

    #bisectionmethod(-2, -1)
    int = intervalSelectioncompa(0, 4)
    print(f'The initial interval was {int}.')
    c = bisectionmethod_compa(2, 3)
    print(f'The final interval with length one [{2}, {3}] contains a root. ' 
          f'The approximate root is {c}.')

    int = intervalSelectioncompb(0, 8)
    print(f'The initial interval was {int}.')
    c = bisectionmethod_compb(6, 7)
    print(f'The final interval with length one [{6}, {7}] contains a root. '
          f'The approximate root is {c}.')

    x_1, count = fpiMethod(1.0, 5.0)
    print(f'The approximation for square root of 5 is {x_1}. '
          f'There are {count} steps needed given my initial guess of 1.')
