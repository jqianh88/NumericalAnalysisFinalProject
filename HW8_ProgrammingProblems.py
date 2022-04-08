import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''

Plot the Adams–Bashforth Three-Step Method approximate solution on [0,1] for
 the differential equation y′ = 1 + y2 and initial condition (a) y0 = 0 (b)
 y0 = 1, along with the exact solution (see Exercise 6.1.7). Use step sizes
 h = 0.1 and 0.05.
'''

# Adams–Bashforth Three-Step Method (third order)
# w_{i+1} = w_i + h [23f_i − 16f_{i−1} + 5f_{i−2}].

#  Program 6.7 Multistep method
#  Inputs: time interval inter,
#  ic=[y0] initial condition, number of steps n,
#  s=number of (multi)steps, e.g. 3 for 3-step method
#  Output: time steps t, solution y
#  Calls a multistep method such as ab2step.m

def diffeq(y, t):               # Differential eq: y' = 1+y^2
    return 1+y**2

# Add taylor expansion for fi_1, fi_2

def exactsol(t, y0):        # Exact sol; y(0) =
    c = np.arctan(y0)
    return np.tan(t+c)

def adams_bashforth3(diffeq, interval, y0, h):
    t = np.arange(interval[0], interval[1] + h, h)
    n = len(t)
    w = np.zeros((n,1))
    w[0] = y0
    #w[1] = solve_ivp(diffeq, [t[0], t[0] + h], w[0], method="RK45").y[0][-1]
    #w[2] = solve_ivp(diffeq, [t[1], t[1] + h], w[1], method="RK45").y[0][-1]
    w[1] = trap_step(diffeq, t[0], w[0], h)      # maybe go back to add diffeq
    w[2] = trap_step(diffeq, t[1], w[1], h)
    fi1 = diffeq(w[2 - 1], t[2 - 1])
    fi2 = diffeq(w[2 - 2], t[2 - 2])
    for i in range(2, n-1):
        fi = diffeq(w[i], t[i])
        w[i+1] = w[i] + (h/12)*(23*fi - 16*fi1 + 5*fi2)
        fi2 = fi1
        fi1 = fi
    return w, t

def trap_step(diffeq, t, x, h):
    z1 = diffeq(x, t)
    g = x + h * z1
    z2 = diffeq(t + h, g)
    return x + h * (z1 + z2) / 2


### Next problem
def secondOdiffeq(firstOdiffeq, y, a, b, t):    # y'' = ay + by'
    return a*y + b* firstOdiffeq

def exactsold2(t):        # Exact sol for problem 2
    return np.exp(3-3*t)

# Solves Aw = b using finite differences method
# h = step size, n = number of steps, bv = boundary value, inter = interval
def approximation(inter, bv, n):
    h = (inter[1] - inter[0]) / (n + 1)  # h = step size
    t = np.linspace(inter[0], inter[1], n + 2)
    b = np.zeros((n+2, 1))  # initialize b vector
    w = np.zeros((n+2, 1)) # initialize solution vector, w
    b[1] = -(1 - h) * bv[0]  # first entry plug in bvec
    b[n] = -(1 + h) * bv[1]  # last entry plug in bvec
    b[0] =  bv[0]    # first entry of solution, w
    b[n+1] = bv[1]  # last entry of solution, w


    # for loop for diagonal
    # ((1-h) * w[i-1]) -2 * w[i] - ((3 * h ** 2) * w[i]) + ((1+h) * w[i+1]) = 0
    A = np.zeros((n+2, n+2))  # allocate matrix
    A[0][0] = 1  # entry 1,1 of the tridiagonal matrix
    A[n + 1][n + 1] = 1  # entry n+1,n+1 of the tridiagonal meatrix
    for i in range(1, n+1):
        A[i][i] = -(2 + 3 * h ** 2)

    # for loop for off diagonals
    for i in range(1, n):           # i = row
        A[i][i + 1] = (1 + h)       # super diagonal
        A[i+1][i] = (1 - h)       # lower diagonal
    print(A)
    print(b)
    #print(b[n + 1])
    w = np.linalg.solve(A, b)
    #print(w[:,0])
    return w[:,0], t


if __name__ == "__main__":
    # h_s
    hs = [0.1, 0.05]
    cs = [0, "pi/4"]
    init_ys = [0,1]      # y_0 to be on same plot
    #y0_exact = [0.0, np.pi/4]
    interval = [0,0.9]
    approxsols = {}
    timesteps = {}
    # Dictionary: key and entry, entries = approx sols and times, need to
    # reference those,  to initialize dict = {}
    for y0 in init_ys:
        for h in hs:
            # returns a list
            approxsols[(y0, h)], timesteps[h] = adams_bashforth3(diffeq,
                                                            interval, y0, h)
    # Plotting
    for plotnum, (y0, c) in enumerate(zip(init_ys, cs)):
        plt.figure(plotnum+1)
        timestepexact = np.linspace(interval[0], interval[1], 1000)
        plt.plot(timestepexact,
                 exactsol(timestepexact, y0),
                 label = f'exactsol:tan(t + {c})')

        for h in hs:
            time = timesteps[h]
            plt.plot(time,
                     approxsols[(y0, h)], 'o',label = f'timestep (h) '
                                                              f'= {h}')
        plt.legend(loc = 'best')
        plt.title(f'Solutions vs Time with y0 = {y0}')
        plt.xlabel('Time')  # Label x-axis
        plt.ylabel('Solutions')  # Label y-axis
        plt.savefig(f'HW8_6.7.5_Initialvalue_{y0}.png', bbox_inches='tight')
    plt.show()


    # Example values:
    inter = [0,1]
    ic = 1
    n = 20
    s = 2


    # For 7.2b
    ns = [9, 19, 39]
    interval_nfd = [0,1]
    bv = [np.exp(3), 1]
    error_array = []
    w_array = []
    w_dict = {}
    x_dict = {}
    error_dict = {}
    exacttime = np.linspace(interval_nfd[0], interval_nfd[1], 1000)
    t_dict = {}


    for n in ns:
        w_dict[n], t_dict[n] = approximation(interval_nfd, bv,
                                             n)
        error_dict[n] = np.abs(w_dict[n] - exactsold2(t_dict[n]))
        print(w_dict[n])
        print(exactsold2(t_dict[n]), 'exact')

    plt.figure(3)
    plt.plot(exacttime, exactsold2(exacttime), 'o', label="Exact Solution: y("
                                                          "t) = "
                                                     "e^(3-3t)")
    for n in ns:
        plt.plot(t_dict[n], w_dict[n], label =f"n = {n}")
        plt.legend(loc='best')
        plt.title(f'Solutions vs Time')
        plt.xlabel('Time')  # Label x-axis
        plt.ylabel('Solutions')  # Label y-axis
    plt.savefig(f'HW8Q2ExactVsApprox_atNs.png', bbox_inches='tight')
    plt.figure(4)
    for n in ns:
        plt.semilogy(t_dict[n][1:n+1],
                     error_dict[n][1:n+1], label =f"n ={n}")
        plt.legend(loc='best')
        plt.title(f'Error vs Time on Semilog Scale')
        plt.xlabel('Time')  # Label x-axis
        plt.ylabel('Error')  # Label y-axis
    plt.savefig(f'HW8CQ2Error_atNs.png', bbox_inches='tight')
    plt.show()




















