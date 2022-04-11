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
    vec = np.array([-6 + 2*(np.sin(T) + 3*np.cos(T)),
           -20 + 6*np.sin(T) + 20*np.cos(T)])
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
    return(w, t)



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
    return(w, t)


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

    # For loop to calculate w_i+1

    for i, time in enumerate(t):
        w_i = w[:, [i]]

        # s1 section
        s1= f(w_i, time, coeff)

        # s2 section
        s2 = f(w_i + (dt/2)*s1, time + (dt/2), coeff)

        # s3 section
        s3 = f(w_i + (dt/2)*s2, time + (dt/2), coeff)

        # s4 section
        s4 = f(w_i + (dt*s3), time + dt, coeff)

        w[:, [i + 1]] = w_i + (dt/6)*(s1 + (2*s2) + (2*s3) + s4)

    t.append(T)                           # Append Final time step

    # w= numpy vector, t = list
    return(w, t)

'''
Adams Moulton Section 
'''
def fmoult(c0, c1, c2, Nfunc):
    f = c0 + c1*Nfunc + c2*Nfunc**2
    return f

def newtons(b0, b1, b2, w):        #w: initial guess
    fw = b0 + b1*w + b2*w**2
    fprime = b1 + 2*b2*w
    z = w - (fw/fprime)
    return z

# Newton's method twice, approximation from 2-3 runs, set up quadratic formula
# - b1 + sqrt(b1^2 - 4ac)/(2b0)
# two quadratic formula values
# newton's - q1, ... newton's -q2 whichever smaller set the solution to the
# one that is closer (smaller difference)

# Adjust tolerance if it takes too long, should not be more than 4 times

def adams_moult3(f, w0, T, numTstep, coeff):        # Make 4th order implicit
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    Ns = np.zeros(numTstep+1)         # initialize N(t)s (1 row array)
    t = [dt * time for time in range(numTstep)]     # t for each timestep

    Ns[0] = coeff[1] - coeff[2]*coeff[3]*w0[1] * w0[0]
    rktemp = RK4(f, w0, 2*dt, 2, coeff)            # 3 initials   2x2

    w[:, 0:3] = rktemp[0]               # ignore last vlue from rk4

    # Beta, A, alpha, x
    c0 = coeff[1]                       #A
    c2 = coeff[3] * coeff[2]            #x*alpha
    c1 = (-coeff[3])*(coeff[2]*w0[0] + w0[1])   # -x*(alpha * N(0) + n(0))

    b2 = (dt / 24)*(9*c2)

    for i, time in enumerate(t[2:]):        # start from i = 2, so shift by 2
        j = i + 2
        Ns[j] = fmoult(c0, c1, c2, w[0, [j]])
        # Define bs
        b0 = w[0, [j]] + (dt / 24) * (9 * c0 +
                                         19 * Ns[j] -
                                         5 * Ns[j-1] +
                                         Ns[j-2])

        b1 = -1 - (dt/24)*(9*c1)

        # Solve Quadratic:
        diff = 1
        approx = w[0, [j]]

        # Loop through until diff < TOL to compute using Newton's
        while diff > 10**-5:
            approx2 = newtons(b0, b1, b2, approx)       # update
            diff = abs(approx2-approx)
            approx = approx2
        w[0, [j+1]] = approx              # Approximate root = next step of i+1
        #print(w[0, [j+1]])
        # replace 149 with q1 or q2 based on whatever you chose, takes away
        # error

        # n(t_i+1) = -alpha*N(t_i+1) + (alpha + Beta)*At_i+1 + alpha*N(0) +n(0)
        w[1, [j+1]] = (-coeff[2] * w[0, [j+1]]) + (coeff[2]+coeff[0])* coeff[
            1]*(time+dt) + coeff[2]* w[0,[0]] + w[1,[0]]

        # (-x)(alpha+B)A*dt
        c1 += (-coeff[3])*(coeff[2]+coeff[0])* coeff[1] * dt

    # Append Final time step
    t.append(T)

    # w= numpy vector, t = list
    return(w, t)




def adams_bash(f, w0, T, numTstep, coeff):    # make 4th order explicit
    # compare
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    #w[:,[0]] = w0                           # initialize w0
    rktemp = RK4(f, w0, 3*dt, 3, coeff)        # 4 initials   2x4

    w[:,0:4] = rktemp[0]

    for i, time in enumerate(t[3:]):
        j = i + 3
        # Initialize ts
        ti = time
        tim1 = time - dt
        tim2 = time - 2 * dt
        tim3 = time - 3 * dt

        # Initialize ws
        wi = w[:, [j]]          #w3  i + 3
        wim1 = w[:, [j-1]]    #w2 i + 3 - 1
        wim2 = w[:, [j-2]]    #w1:i + 3 - 2
        wim3 = w[:, [j-3]]    #w0 i+3-3


        # Adams Bashforth: wim1 = wi minus 1
        wip1 = w[:, [j]] + ((dt / 24) * (55 * f(wi, ti, coeff) -
                                         59 * f(wim1, tim1, coeff) +
                                         37 * f(wim2, tim2, coeff) -
                                         9 * f(wim3, tim3, coeff)))
        w[:, [j + 1]] = wip1  # Store in w

    t.append(T)                                       # Append Final time step

    # w= numpy vector, t = list
    return(w, t)


'''
Three step method:
# Runge kutta: use RK4 so that orders are correct
1) runge kutta -- Initialization
2) adams bashforth (explicit) Predictor
3) Adams Moulton (implicit) Corrector  
'''

def adams_pcec(f, w0, T, numTstep, coeff):
    dt = T/numTstep                                      # delta t = step size
    w = np.zeros((w0.size, numTstep+1))                 # initialize w sol
    t = [dt * time for time in range(numTstep)]     # t for each timestep
    rktemp = RK4(f, w0, 3*dt, 3, coeff)        # 4 initials   2x4

    w[:,0:4] = rktemp[0]

    for i, time in enumerate(t[3:]):
        j = i + 3
        # Initialize ts
        ti = time
        tim1 = time - dt
        tim2 = time - 2 * dt
        tim3 = time - 3 * dt

        # Initialize ws
        wi = w[:, [j]]  # w3  i + 3
        wim1 = w[:, [j - 1]]  # w2 i + 3 - 1
        wim2 = w[:, [j - 2]]  # w1:i + 3 - 2
        wim3 = w[:, [j - 3]]  # w0 i+3-3

        # Adams Bashforth: wim1 = wi minus 1
        wip1p = w[:, [j]] + ((dt / 24) * (55 * f(wi, ti, coeff) -
                                         59 * f(wim1, tim1, coeff) +
                                         37 * f(wim2, tim2, coeff) -
                                         9 * f(wim3, tim3, coeff)))





        tip1 = time + dt



        # Adams Moulton: wip1 = wi plus 1
        wip1c = w[:, [j]] + (dt/24)*(9*f(wip1p, tip1, coeff) +
                                    19*f(wi, ti, coeff) -
                                    5*f(wim1, tim1, coeff) +
                                    f(wim2, tim2, coeff))
        w[:, [j+1]] = wip1c               # Store in w, next step

    t.append(T)

    # w= numpy vector, t = list
    return(w, t)



def avg_error(array1, array2):
    error = np.linalg.norm(array1-array2, axis = 0)

    return np.mean(error)

def avg_percent_error(actual, expected):
    exp = np.linalg.norm(expected, axis = 0)               # expected
    error = np.linalg.norm(actual-expected, axis = 0)      # based on norm
    avg_per = np.divide(error, exp)                        # Avg percent error

    return np.mean(avg_per)




if __name__ == '__main__':

    '''
    Tests: Do not Touch!!!!!!!!
    '''
    # Inputs for Error Checking:
    run_tests = True

    test_w0 = np.array([0., 0.]).reshape((2,1))   # IC column vec
    test_T = 100
    test_numTsteplist = [10**3, 10**4, 10**5, 10**6]
    test_coeff = (-6, 2, -20, 6)


    method_list = [euler, comp_trap, RK4, adams_bash, adams_pcec]
    err_list = [[] for _ in method_list]

    if run_tests:

        for i, method in enumerate(method_list):
            for j, test_numTstep in enumerate(test_numTsteplist):
                print(i, j, 'ij')
                array1, test_timeslist = method(ftest, test_w0, test_T,
                                               test_numTstep, test_coeff)

                array2 = ftestanswer(test_timeslist)

                err_list[i].append(avg_error(array1, array2))


        methodnames = ['Euler', 'Composite Trapezoid', 'RK4',
                       'Adams Bashforth', 'Predictor Corrector']

        plt.figure(1)
        plt.clf()  # Clear the plot
        for i, name in enumerate(methodnames):
            plt.loglog(test_numTsteplist, err_list[i], label=f"{name}")
            plt.legend(loc='best', prop={'size': 6})
            plt.title(f'Error vs N of All Methods')
            plt.xlabel('N (Number of Steps)')  # Label x-axis
            plt.ylabel('Error')  # Label y-axis
            plt.savefig(f'Math361FinalProjectTestSetError{name}.png',
             bbox_inches='tight')
        plt.show()



    # Part 1 Inputs

    alpha = 10 ** 4  # alpha: number of fragments that exceed a few gs
    Beta = 70  # Beta: number of primary fragments
    x = 3 * 10 ** (-10)  # Constant
    A = 100  # num of satellites launched - num re-entering
    N0 = 2 * 10 ** 3  # N(0), the IC for number of satellites
    n0 = 5 * 10 ** 4  # n(0), the number of fragments
    w0 = np.array([N0, n0]).reshape((2, 1))  # IC column vec
    T = 100
    numTstep = 100          #play with timestep, which better for presentation
    numTstepList = [1000, 10000, 10 ** 5, 10 ** 6]
    coeff = (Beta, A, alpha, x)

    run_prints = True
    method_list = [euler, comp_trap, RK4, adams_bash, adams_pcec]
    methodnames = ['Euler', 'Composite Trapezoid', 'RK4', 'Adams ' 
                'Bashforth','Predictor Corrector']
    func = f         # Use this to change for advanced and add to gnames
    graphnames = ['Satellites', 'Fragments']
    sol_list = [[] for _ in method_list]
    times_list = [[] for _ in method_list]
    per_err_list = [[] for _ in method_list]

    if run_prints:


        # Get Data
        for i, method in enumerate(method_list):
            for j, numTstep in enumerate(numTstepList):
                print(i,j, 'ij')
                # Expected Results: Based on RK4
                rk4_sol, time = RK4(func, w0, T, numTstep, coeff)

                # Actual Results
                z, time = method(func, w0, T, numTstep, coeff)
                sol_list[i].append(z)
                times_list[j] = time

        # Get Errors: based on RK4, keep track of RK4
                per_err_list[i].append(avg_percent_error(z, rk4_sol))



        # Plotting RK4:
        exprk4, trk4 = RK4(func, w0, T, 10 ** 3, coeff)
        for i, graph in enumerate(graphnames):
            plt.figure(i+1)
            plt.clf()
            plt.plot(trk4, exprk4[i])
            plt.xlabel('Time (Years)')  # Label x-axis
            plt.ylabel(f'{graph}')  # Label y-axis
            plt.title(f'RK4 Expected {graph} with N = 10^3')
            plt.savefig(f'Math361FinalProjectRK4Expected{graph}Simple.png',
            bbox_inches='tight')


        # Plot Percent Error
        plt.figure(4)
        plt.clf()  # Clear the plot
        for i, name in enumerate(methodnames):
            plt.loglog(numTstepList, per_err_list[i], label=f"{name}")
            plt.legend(loc='best', prop = {'size': 6})
        plt.title(f'Percent Error vs N with RK4 as Expected Result')
        plt.xlabel('N (Number of Steps)')  # Label x-axis
        plt.ylabel('Percent Error')  # Label y-axis
        plt.savefig(f'Math361FinalProjectPercentErrorRk4baseSimple.png',
                    bbox_inches='tight')


        methodnames = ['Euler', 'Composite Trapezoid', 'RK4', 'Adams ' 
                'Bashforth','Predictor Corrector']

        # Plotting:
        for k, graph in enumerate(graphnames):
            plt.figure(k+5)
            for j, name in enumerate(methodnames):
                plt.plot(times_list[k], sol_list[j][k][0], label= f' {name}')
                plt.title(f'{graph} vs Time with Each Method')
                plt.xlabel('Time (Years)')  # Label x-axis
                plt.ylabel(f'{graph}')  # Label y-axis
                plt.legend(loc='best')
            plt.savefig(f'Math361FinalProject{graph}MethodTimeSimple.png',
            bbox_inches='tight')
        plt.show()











    '''
    Section 2: Plotting 
    '''
'''
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
    row_sat = [[] for Nvalue in numTstepList]  # Space allo for list of lists
    row_frag = [[] for step in numTstepList]  # Space allo for list of lists

    print(row_sat)
    # Space allocation for each method: satellites solution
    xs_sat = [row_sat for method in method_list]
    print(xs_sat)
    # Space allocation for each method: Fragments solution
    xs_frag = [row_frag for method in method_list]


    # Space allocation for the list of lists of timesteps
    ts = [[] for Nvalue in numTstepList]
    print(ts)

    for i, method in enumerate(method_list):
        for j, numTstep in enumerate(numTstepList):
            # Extract approximation and the timestep for the method in methlist
            xs_sat[i][j], xs_frag[i][j], ts[j] = method(f, w0, T, numTstep,
                                                        coeff)

    # For loop to increment plot number and access Method name to plot
    #print(xs)

# Alternative attempt
    for plotnum, (time, numTstep) in enumerate(zip(
                         ts, numTstepList )):
        plt.figure(plotnum + 1)
        for i,  (methName, sol_sat, sol_frag) in enumerate(zip(methodnames,
                                                               xs_sat,
                                                               xs_frag)):
            plt.figure(plotnum + 1)
            plt.plot(ts[plotnum], sol_sat[plotnum], 'o', label=
            f"Satellite{methName}")
            plt.plot(ts[plotnum], sol_frag[plotnum], label=
            f"Debris{methName}")
            plt.legend(loc='upper left')
            plt.title(f'Approximation vs Time with {numTstep} steps and '
                      f'{T/numTstep} step size')
            plt.xlabel('Time')  # Label x-axis
            plt.ylabel('Approximation')  # Label y-axis
    #plt.show()
'''

# Attempt
'''
    for plotnum, (methName, sol_sat, sol_frag) in enumerate(zip(methodnames,
                                                            xs_sat, xs_frag)):
        
        #print(methName, sol_sat, 'here')
        #print(methName,  sol_frag, 'here2')
        plt.figure(plotnum+1)
        plt.plot(ts[plotnum], sol_sat[plotnum], 'o', label =
        f"Satellite{methName}")
        plt.plot(ts[plotnum], sol_frag[plotnum], label=
        f"Debris{methName}")
        plt.legend(loc='upper left')
        plt.title(f'Approximation vs Time using {methName} Method')
        plt.xlabel('Time')  # Label x-axis
        plt.ylabel('Approximation')  # Label y-axis
    plt.show()
'''


# Error plotting
# 2 types: 1) vs euler with super fine step size, 2) vs given tpoints from pap
'''
    # Reference 
    ref_sol = euler(f, w0, 500, 10**10, coeff), 'euler')
    
    # Alternate Version
    row_sat_er = [[] for Nvalue in numTstepList]  # Space allo for list of lists
    row_frag_er = [[] for step in numTstepList]  # Space allo for list of lists

    #print(row)
    # Space allocation for each method: satellites solution
    xs_sat_er = [row_sat_er for method in method_list]

    # Space allocation for each method: Fragments solution
    xs_frag_er = [row_frag_er for method in method_list]
    
    # Space allocation for list of errors for each N and method
    errorlist = [[0.0 for nvalue in numTstepList] for method in method_list]

    # Space allocation for the list of lists of timesteps
    ts = [[] for Nvalue in numTstepList]

    # For loop to calculate the error between the approximation and exact sol
    for i, method in enumerate(method_list):
        for j, numTstep in enumerate(numTstepList):
            # Extract approximation and the timestep for the method in methlist
            (xs[i][j], ts[j]) = method(f, w0, T, numTstep, coeff)

            # Calculate error and assign to index in the list
            errorlist[i][j] = abs(refsol - xs[i][j][-1])

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
    
    
    
    
# writing individual plots
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