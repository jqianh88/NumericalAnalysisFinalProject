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

# Make f test base on
#solve[{x' = -6sin(t) + 2cos(t), y' = -20sin(t) + 6cos(t)}]

def ftest(w, t, coeff):

    # Differential equation: dw/dt = cos(wt)
    a, b, c, d = coeff          # Unpack coefficients
    x,y = w
    vec = np.array([a*np.sin(t) + b*np.cos(t), c*np.sin(t) + d*np.cos(
        t)]).reshape((2,1))
    return vec


# Solution to test: with
# x(t) = c_1 + 2 (sin(t) + 3 cos(t)) w/ t=0 --> -6 + 2 (sin(t) + 3 cos(t))
# y(t) = c_2 + 6 sin(t) + 20 cos(t) w/ t=0 --> -20 + 6sin(t) + 20cos(t)
def ftestanswer(T):
    vec = np.array([-6 + 2*(np.sin(T) + 3*np.cos(T)), -20 + 6*np.sin(T) +
                    20*np.cos(
        T)]).reshape((2,1))
    return vec
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
def euler(ftest, w0, T, numTstep, coeff):
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
                              f(w_i + dt*f(w_i, time, coeff), time + dt,
                                 coeff))
        w[:, [i + 1]] = wnext                       # Store in w

    t.append(T)                                       # Append Final time step
    # w= numpy vector, t = list
    return (w, t)



'''def adams_moult3(f, w0, T, numTstep, coeff):        # Make 4th order implicit
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    w[:,[0]] = w0                           # initialize w0
    rktemp = RK4(f, w0, T, 2, coeff)            # 3 initials   2x2
    #print(rktemp)
    w[:, 0:3] = rktemp[0]
    #print(w[:, 0:3], '0-3')
    #print(t)
    for i, time in enumerate(t[2:]):
        # Initialize ts
        ti = time
        tim1 = time - dt
        tim2 = time -2*dt
        tip1 = time + dt

        # Initialize ws
        wi = w[:, [i+2]]
        wim1 = w[:, [i+2-1]]
        wim2 = w[:, [i+2-2]]
        wip1 = w[:, [i+2+1]]

        # Adams Moulton: wip1 = wi plus 1
        wip1 = wi + (dt / 24) * (9 * ftest(wip1, tip1, coeff) +
                                         19 * ftest(wi, ti, coeff) -
                                         5 * ftest(wim1, tim1, coeff) +
                                         ftest(wim2, tim2, coeff))
        w[:, [i + 2+1]] = wip1  # Store in w

    t.append(T)                                       # Append Final time step
    # w= numpy vector, t = list
    return (w, t)
'''


"""
RK4: fourth order Runge-Kutta

This function performs Runge-Kutta 4 to solve the pair of ODEs. 
Inputs:
Outputs: Two lists are output, x as a list of the displacement values at each 
time step and t, a list of the time corresponding to the values of x. 


Observations: needs small time step otherwise it blows up
"""

def RK4(f, w0, T, numTstep, coeff):
    dt = T/numTstep                                # delta t = step size = h
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    w[:,[0]] = w0                           # initialize w0
    #print(w[:,[0]])
    # For loop to calculate w_i+1

    for i, time in enumerate(t):
        w_i = w[:, [i]]
        #print(w_i, 'i')
        # s1 section
        s1= f(w_i, time, coeff)
        #print(s1, 's1')
        # s2 section
        s2 = f(w_i + (dt/2)*s1, time + (dt/2), coeff)
        #print(s2, 's2')

        # s3 section
        s3 = f(w_i + (dt/2)*s2, time + (dt/2), coeff)
        #print(s3, 's3')
        # s4 section
        s4 = f(w_i + (dt*s3), time + dt, coeff)
        #print(s4, 's4')

        w[:, [i + 1]] = w_i + (dt/6)*(s1 + (2*s2) + (2*s3) + s4)
        #print(w[:, [i + 1]], 'i+1')
    t.append(T)                           # Append Final time step

    return (w, t)


def adams_bash(f, w0, T, numTstep, coeff):    # make 4th order explicit
    # compare
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    #w[:,[0]] = w0                           # initialize w0
    rktemp = RK4(f, w0, T, 3, coeff)        # 4 initials   2x4
    print(rktemp)
    w[:,0:4] = rktemp[0]
    print(w[:,[0]], 'here')
    for i, time in enumerate(t[3:]):
        # Initialize ts
        ti = time
        tim1 = time - dt
        tim2 = time - 2 * dt
        tim3 = time - 3 * dt
        print(ti, tim1, tim2, tim3, 'ts')
        # Initialize ws
        wi = w[:, [i + 3]]          #w3  i + 3
        wim1 = w[:, [i + 2]]    #w2 i + 3 - 1
        wim2 = w[:, [i + 1]]    #w1:i + 3 - 2
        wim3 = w[:, [i]]    #w0 i+3-3
        print(wi, wim1, wim2, wim3, 'ws')

        # Adams Bashforth: wim1 = wi minus 1
        wip1 = w[:, [i+3]] + ((dt / 24) * (55 * f(wi, ti, coeff) -
                                         59 * f(wim1, tim1, coeff) +
                                         37 * f(wim2, tim2, coeff) -
                                         9 * f(wim3, tim3, coeff)))
        w[:, [i + 3 + 1]] = wip1  # Store in w

    t.append(T)                                       # Append Final time step
    # w= numpy vector, t = list
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
    #print('w', w)
    for i, time in enumerate(t[3:]):
        # Initialize ts
        ti = time
        tim1 = time - dt
        tim2 = time -2*dt
        tim3 = time - 3*dt
        tip1 = time + dt
        #print(ti, tim1, tim2, tim3, tip1)
        # Initialize ws
        wi = w[:, [i+3]]
        wim1 = w[:, [i+3-1]]
        wim2 = w[:, [i+3-2]]
        wim3 = w[:, [i+3-3]]
        #print(wi, wim1, wim2, wim3)

        # Adams Bashforth: wim1 = wi minus 1
        wip1p = w[:, [i+3]] + (dt/24)*(55*ftest(wi, ti, coeff) -
                                    59*ftest(wim1, tim1, coeff) +
                                    37*ftest(wim2, tim2, coeff) -
                                    9*ftest(wim3, tim3, coeff))

        # Adams Moulton: wip1 = wi plus 1
        wip1c = w[:, [i+3]] + (dt/24)*(9*ftest(wip1p, tip1, coeff) +
                                    19*ftest(wi, ti, coeff) -
                                    5*ftest(wim1, tim1, coeff) +
                                    ftest(wim2, tim2, coeff))
        w[:, [i + 3 +1]] = wip1c               # Store in w, next step
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
    T = 300
    numTstep = 1000
    numTstepList = [100, 1000, 10 ** 4, 10 ** 5]
    #numTstepList = [10**4, 10**5]
    coeff = (Beta, A, alpha, x)

    print(euler(f, w0, T, numTstep, coeff), 'euler')  # 126723.69846774
    print(comp_trap(f, w0, T, numTstep, coeff), 'comptrap')   # 126767.24327992
    print(RK4(f, w0, T, numTstep, coeff), 'Rk4')     # 126767.21446085
    #print(adams_moult3(f, w0, T, numTstep, coeff), 'moul3')  # 206092.29727389
    print(adams_bash(f, w0, T, numTstep, coeff), 'bash')  #  126783.3287603
    #print(adams_pcec(f, w0, T, numTstep, coeff), 'pcec')    # 211923.96086034
    #print(ftestanswer(T))

    # Inputs for Error Checking

    '''    
    a = -6
    b = 2
    c = -20
    d = 6
    X0 = -6              # N(0), the IC for number of satellites
    Y0 = -20              # n(0), the number of fragments
    w0 = np.array([X0, Y0]).reshape((2,1))   # IC column vec
    T = 10
    numTstep = 100
    coeff = (a, b, c, d)
    '''

    #print(euler(ftest, w0, T, numTstep, coeff), 'euler') #- verified
    #print(comp_trap(f, w0, T, numTstep, coeff), 'comptrap') #- verified
    #print(RK4(f, w0, T, numTstep, coeff)[0], 'Rk4') # -verified
    #print(adams_moult3(f, w0, T, numTstep, coeff), 'moul3')  - wrong
    #print(adams_bash(f, w0, T, numTstep, coeff), 'bash')  #  126783.3287603
    #print(adams_pcec(f, w0, T, numTstep, coeff), 'pcec')    # 211923.96086034
    #print(ftestanswer(T))


    '''
    Section 2: Plotting 
    '''
    #approx, ts = euler(f, w0, T, numTstep, coeff)
    #print(approx[0], 'approx')
    #plt.plot(ts, approx[0], label = "Euler")
    #plt.show()

    #approx, ts = euler(f, w0, T, numTstep, coeff)
    #approx1, ts1 = comp_trap(f, w0, T, numTstep, coeff)
    #print(approx[0], 'approx')
    #print(approx1[0], 'comptrap')
    #plt.plot(ts, approx[0], label = "comp_trap")
    #plt.show()

    # List of Method Functions
    #method_list = [euler, comp_trap]
    method_list = [euler, comp_trap, RK4, adams_bash]
    #method_list = [euler, comp_trap, RK4, adams_bash, adams_pcec]

    # Names of Method Functions:
    #methodnames = ['Euler', 'Composite Trapezoid']
    methodnames = ['Euler', 'Composite Trapezoid',
                   'RK4','4th Order Adams Bashforth',]
    #methodnames = ['Euler', 'Composite Trapezoid', 'RK4',
    # '4th Order Adams Bashforth','Adams Bashforth Moulton Predictor Corrector']

    # Alternate Version
    row = [[] for Nvalue in numTstepList]  # Space allocation for list of lists
    xs = [row for method in method_list]  # Space allocation for each method

    # Space allocation for the list of lists of timesteps
    ts = [[] for Nvalue in numTstepList]

    for i, method in enumerate(method_list):
        for j, N in enumerate(numTstepList):
            # Extract approximation and the timestep for the method in methlist
            (xs[i][j], ts[j]) = method(f, w0, T, numTstep, coeff)
            #print(xs[i][j][0], xs[i][j][1], ts[j], 'gothere')

    #print(ts[0], 'here')
    # For loop to increment plot number and access Method name to plot
    print(xs)

    for plotnum, (methName, sol) in enumerate(zip(methodnames, xs)):
        print(methName, sol, 'here')
        #plt.figure(plotnum+1)
        #plt.plot(ts[plotnum], xs[plotnum][plotnum][0], 'o', label =
        #f"Satellite{methName}")
        #plt.plot(ts[plotnum], xs[plotnum][plotnum][1], label=
        #f"Debris{methName}")
        #plt.legend(loc='upper left')
        #plt.title(f'Approximation vs Time using {methName} Method')
        #plt.xlabel('Time')  # Label x-axis
        #plt.ylabel('Approximation')  # Label y-axis
    #plt.show()




# Errror plotting
'''
    # Space allocation for list of errors for each N and method
    errorlist = [[0.0 for nvalue in numTstepList] for method in method_list]

    # Space allocation for the list of lists of timesteps
    ts = [[] for Nvalue in numTstepList]

    # For loop to calculate the error between the approximation and exact sol
    for i, method in enumerate(method_list):
        for j, N in enumerate(numTstepList):
            # Extract approximation and the timestep for the method in methlist
            (xs[i][j], ts[j]) = method(f, w0, T, numTstep, coeff)

            # Calculate error and assign to index in the list
            errorlist[i][j] = abs(exactsol - xs[i][j][-1])

    # For loop to increment plot number and access Method name to plot
    for plotnum, methName in enumerate(methodnames):
        plt.figure(plotnum + 1)
        plt.clf()  # Clear the plot
        plt.loglog(numTstepList, errorlist[i], label=f"{methName}")
        plt.legend(loc='upper left')
        plt.title(f'Error vs N using {methName} Method')
        plt.xlabel('N')  # Label x-axis
        plt.ylabel('Error')  # Label y-axis
        plt.savefig(f'Math361FinalProject{methName}.png', bbox_inches='tight')

    # plt.show()
'''















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