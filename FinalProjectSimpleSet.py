import numpy as np
import matplotlib.pyplot as plt
import math

def f(w, t, coeff):
    # Differential equation: dN/dt = A-xnN,
    # Differential equation: dn/dt = BA + axnN
    Beta, A, alpha, x = coeff          # Unpack coefficients
    N, n = w                    # unpack Vector
    vec = np.array([A-x*n*N, Beta*A + alpha*x*n*N]).reshape((2,1))
    return vec

def fadvanced(w, t, coeff):         # advanced equations vec of 5 instead of 2
    pass

# Make f test based on spring system
def ftest(w, t, coeff):
    pass

# just apply from before
def ftestanswer(w,t, coeff):
    pass
# look at errors, order
# how generated method, figures: time complexity, error,
# how is the method on new data and how it compares to old data

'''
This function performs Forward Euler to solve the simultaneous ODE. 
Inputs: 
    - ICvec: column vector of the initial conditions
    - T:  final time to solve until 
    - numTStep: number of timesteps to use while solving 
'''
def euler(f, w0, T, numTstep, coeff):
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    w[:,[0]] = w0                           # initialize w0

    # For loop to calculate w_i+1
    for i, time in enumerate(t):            # i = index, time = value
        w_i = w[:, [i]]                         # vector w at current timestep
        wnext = w_i + dt*f(w_i, time, coeff)           # FE step
        w[:, [i + 1]] = wnext                       # Store in w

    t.append(T)                                       # Append Final time step
    # w= numpy vector, t = list
    return (w, t)



def comp_trap(f, w0, T, numTstep, coeff):
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    w[:,[0]] = w0                           # initialize w0

    for i, time in enumerate(t):
        w_i = w[:, [i]]
        wnext = w_i + (dt/2)*(f(w_i, time, coeff) +
                              f(w_i + dt*f(w_i, time, coeff), time + dt, coeff))
        w[:, [i + 1]] = wnext                       # Store in w

    t.append(T)                                       # Append Final time step
    # w= numpy vector, t = list
    return (w, t)



def adams_moult2(f, w0, T, numTstep, coeff):        # Make 4th order implicit
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    w[:,[0]] = w0                           # initialize w0
    rktemp = RK4(f, w0, T, 1, coeff)            # 2 initials   2x2
    #print(rktemp)
    w[:, 0:2] = rktemp[0]
    #print(w[:, 0:2], '0-2')
    print(t)
    for i, time in enumerate(t[2:]):
        # Initialize ts
        ti = time
        tim1 = time - dt
        tim2 = time -2*dt
        tip1 = time + dt

        # Initialize ws
        wi = w[:, [i]]
        wim1 = w[:, [i-1]]
        wim2 = w[:, [i-2]]
        wip1 = w[:, [i+1]]

        # Adams Moulton: wip1 = wi plus 1
        wip1 = w[:, [i]] + (dt / 24) * (9 * f(wip1, tip1, coeff) +
                                         19 * f(wi, ti, coeff) -
                                         5 * f(wim1, tim1, coeff) +
                                         f(wim2, tim2, coeff))
        w[:, [i + 1]] = wip1  # Store in w

    t.append(T)                                       # Append Final time step
    # w= numpy vector, t = list
    return (w, t)



def adams_bash(f, w0, T, numTstep, coeff):    # make 4th order explicit compare
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    w[:,[0]] = w0                           # initialize w0

    for i, time in enumerate(t):
        # Initialize ts
        ti = time
        tim1 = time - dt
        tim2 = time - 2 * dt
        tim3 = time - 3 * dt

        # Initialize ws
        wi = w[:, [i]]
        wim1 = w[:, [i - 1]]
        wim2 = w[:, [i - 2]]
        wim3 = w[:, [i - 3]]

        # Adams Bashforth: wim1 = wi minus 1
        wip1 = w[:, [i]] + (dt / 24) * (55 * f(wi, ti, coeff) -
                                         59 * f(wim1, tim1, coeff) +
                                         37 * f(wim2, tim2, coeff) -
                                         9 * f(wim3, tim3, coeff))
        w[:, [i + 1]] = wip1  # Store in w

    t.append(T)                                       # Append Final time step
    # w= numpy vector, t = list
    return (w, t)






"""
RK4: fourth order Runge-Kutta

This function performs Runge-Kutta 4 to solve the pair of ODEs. 
Inputs: There are 7 input values: w0, z, m, w, x0, T, and N which are the 
natural frequency, damping ratio, mass of the system, the forcing frequency, 
the initial condition (column vector), the final time to solve until, and the 
number of timesteps to use while solving, respectively. 
Outputs: Two lists are output, x as a list of the displacement values at each 
time step and t, a list of the time corresponding to the values of x. 
"""

def RK4(f, w0, T, numTstep, coeff):
    dt = T/numTstep                                # delta t = step size = h
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    w[:,[0]] = w0                           # initialize w0

    # For loop to calculate w_i+1

    for i, time in enumerate(t):
        w_i = w[:, [i]]

        # s1 section
        s1= f(w_i, time, coeff)

        # s2 section
        s2 = f(w_i + ((dt/2)*s1), time + (dt/2), coeff)


        # s3 section
        s3 = f(w_i + ((dt/2)*s2), time + (dt/2), coeff)

        # s4 section
        s4 = f(w_i + (dt*s3), time + dt, coeff)


        w[:, [i + 1]] = w_i + (dt/6)*(s1 + (2*s2) + (2*s3) + s4)

    t.append(T)                           # Append Final time step

    return (w, t)



'''
Three step method:
# Runge kutta: use RK4 so that orders are correct
1) runge kutta -- Initialization
2) adams bashforth (explicit) Predictor
3) Adams Moulton (implicit) Corrector  
'''


def adams_pcec(f, w0, T, numTstep, coeff):
    dt = T/numTstep                                # delta t = step size = h
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    rktemp = RK4(f, w0, T, 3, coeff)        # 4 initials   2x4
    #print(rktemp)
    w[:,0:4] = rktemp[0]
    #print(w[:,0:4])                 # clean up 111-114
    for i, time in enumerate(t[4:]):
        # Initialize ts
        ti = time
        tim1 = time - dt
        tim2 = time -2*dt
        tim3 = time - 3*dt
        tip1 = time + dt

        # Initialize ws
        wi = w[:, [i+4]]
        wim1 = w[:, [i+4-1]]
        wim2 = w[:, [i+4-2]]
        wim3 = w[:, [i+4-3]]

        # Adams Bashforth: wim1 = wi minus 1
        wip1p = w[:, [i+4]] + (dt/24)*(55*f(wi, ti, coeff) -
                                    59*f(wim1, tim1, coeff) +
                                    37*f(wim2, tim2, coeff) -
                                    9*f(wim3, tim3, coeff))

        # Adams Moulton: wip1 = wi plus 1
        wip1c = w[:, [i+4]] + (dt/24)*(9*f(wip1p, tip1, coeff) +
                                    19*f(wi, ti, coeff) -
                                    5*f(wim1, tim1, coeff) +
                                    f(wim2, tim2, coeff))
        w[:, [i +4 + 1]] = wip1c                       # Store in w
    return (w, t)










if __name__ == '__main__':

    # Part 1 Inputs
    alpha = 10**4           # alpha: number of fragments that exceed a few gs
    Beta = 70                    # Beta: number of primary fragments
    x = 3*10**(-10)           # Constant
    A = 100                   # num of satellites launched - num re-entering
    N0 = 2*10**3              # N(0), the IC for number of satellites
    n0 = 5*10**4              # n(0), the number of fragments
    w0 = np.array([N0, n0]).reshape((2,1))   # IC column vec
    T = 10
    numTstep = 100
    coeff = (Beta, A, alpha, x)
    #print(euler(f, w0, T, numTstep, coeff), 'euler')  # 126723.69846774
    #print(comp_trap(f, w0, T, numTstep, coeff), 'comptrap')   # 126767.24327992
    #print(RK4(f, w0, T, numTstep, coeff), 'Rk4')     # 126767.21446085
    print(adams_moult2(f, w0, T, numTstep, coeff), 'moul2')  # 124210.85982367
    #print(adams_bash(f, w0, T, numTstep, coeff), 'bash')  #  126783.3287603
    print(adams_pcec(f, w0, T, numTstep, coeff), 'pcec')    # 126783.3287603




''' 
     Make it specific to the problem 
     write as a system instead of generalized 
     
     100222.75568076
     
     
     next steps: 
     find an example 
     check all algorithms with example with test f
     
     
     Python timing routines 
     location error 
     call it once (tells time)
     call second time(tells time) 
     look at difference 
     look at n 
     plot n vs time 
     time diagrams 
     
     nlogn or n^2 check 
     
     Tlk about: 
     Order of error
     Time complexity 
     Using unoptimized right now so look at the corresponding 
     - run timing things [may not be an issue but include in discussion]
     
     Change N: look at log plots 
     
     1) analyze predictor corrector  (analysis of the method), use mass 
     spring system for 
     2) replicating data that someone has done 
     3) seeing how it changes with 
     
     For predictor corrector method 
     ****Find a set of diffeq that you know error 
     *** So you know error and timing 
     
     Then talk about how you applied this method to eqs of interest and how 
     they match up
     
     
     
     Euler, super fine spacing and use as expected results because stable at 
     super small step size , then compare to 
     graph of adams predictor corrector 
     
     
     
     show apc 
     new data apc 
     generate expeted results from euler with super fine steps and store data
     compare how apc did to that 
     
     
     talk about how other methods may have helped 
     
     think about how you will present this in 3 slides 
     1) O(h) graphs and O(n) graphs for APC (error and timing, respectively)
     - compare to FE which is order h--> predictor, then corrector, and how 
     to gether they work and how they improved 
     - euler, ab, am, then apc  
     2) two pics: solid line of FE expected results, more and more 
     complicated stuff to show error and complexity, make determination 
     of which one was better, dotted APC
     - how different algorthims work with Syst equations 
     - applied to interesting probelm 
     3) Conclusions
        - could have optimized if didn't do so many function calls
        - # of function calls = how much faster could run 
        - just save previous function calls 
        - mention this, save time but expense of memory, 


*** Why are you working on what you are working 

For fun: 
1) solver class
2) equations class 

refactor --> object oriented 

'''