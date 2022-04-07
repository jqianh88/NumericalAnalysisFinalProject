import math
#import numpy
#import pandas
import matplotlib.pyplot as plt

'''
def newtonsMethod_a_previous(x_i):
    iter_count = 0
    pk_list = []
    sequence_alpha = []
    f_double = 2 * math.exp(1 - 1) - 2
    f_triple = 2 * math.exp(1 - 1)
    x_ip1 = 0
    x_ip2 = 0
    ak = 1
    for val in range(100):
        ak2 = ak
        x_ip3 = x_ip2
        x_ip2 = x_ip1
        f = 2 * math.exp(x_i - 1) - x_i ** 2 - 1
        f_prime = 2 * math.exp(x_i - 1) - 2 * x_i
        x_ip1 = x_i - (f/f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 3 and pk_list[val] != pk_list[val-1]:
            #ak = math.log10(abs(x_ip1 - x_ip2)/abs(x_ip2 - x_ip3) if abs(x_ip1 - x_ip2)/abs(x_ip2 - x_ip3) > 0 else 1)
            ak = math.log10(
                abs(pk_list[val] - pk_list[val-1]) / abs(pk_list[val-1] - pk_list[val-2]) if abs(pk_list[val] - pk_list[val-1]) / abs(pk_list[val-1] - pk_list[val-2]) >0 else 1)
            print(ak)
            ak2 = math.log10(
                abs(pk_list[val-1] - pk_list[val - 2]) / abs(pk_list[val - 2] - pk_list[val - 3]) if abs(pk_list[val-1] - pk_list[val - 2]) / abs(pk_list[val - 2] - pk_list[val - 3]) > 0 else 1)
            print(ak2)
            alpha = ak/ak2
            print('alpha', alpha)
            sequence_alpha.append(alpha if ak!= 0 else 'converged')
            print(sequence_alpha)
            print(len(sequence_alpha))
            if x_ip1 == x_i:
                print('finish')
                forward_error = abs(x_ip1-1)
                backward_error = 2 * math.exp(x_ip1 - 1) - x_ip1 ** 2 - 1
                f_r = 2 * math.exp(1 - 1) - 1 ** 2 - 1
                f_prime_r = 2 * math.exp(1 - 1) - 2 * 1
                iter_count += 1
                print(sequence_alpha)
                print('f_r', f_r)
                return (f_r, f_prime_r, f_double, f_triple, iter_count, x_ip1, pk_list, sequence_alpha, forward_error, backward_error)
        x_i = x_ip1
        iter_count += 1
        
        
def modifiedNM_a_previous(x_i):
    iter_count = 0
    pk_list = []
    sequence_alpha = []
    x_ip1 = 0
    x_ip2 = 1
    ak = 1
    for val in range(100):
        ak2 = ak
        x_ip3 = x_ip2
        x_ip2 = x_ip1
        f = 2 * math.exp(x_i - 1) - x_i ** 2 - 1
        f_prime = 2 * math.exp(x_i - 1) - 2 * x_i
        x_ip1 = x_i - (3*f/f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 1:
            ak = math.log10(abs(x_ip1 - x_ip2)/abs(x_ip2 - x_ip3) if abs(x_ip1 - x_ip2)/abs(x_ip2 - x_ip3) > 0 else 1)
            alpha = ak/ak2
            sequence_alpha.append(alpha if ak != 0 else 'converged')
            if x_ip1 == x_i:
                forward_error = abs(x_ip1-1)
                backward_error = 2 * math.exp(x_ip1 - 1) - x_ip1 ** 2 - 1
                iter_count += 1
                return (iter_count, x_ip1, pk_list, sequence_alpha, forward_error, backward_error)
        x_i = x_ip1
        iter_count += 1
        
def newtonsMethod_b_previous(x_i):
    iter_count = 0
    pk_list = []
    sequence_alpha = []
    f_double = -1 / (3 - 2) ** 2
    f_triple = 2 / (3 - 2) ** 2
    x_ip1 = 0
    x_ip2 = 1
    ak = 1
    for val in range(100):
        ak2 = ak
        x_ip3 = x_ip2
        x_ip2 = x_ip1
        f = math.log(3 - x_i) + x_i - 2
        f_prime = -(x_i - 2) / (3 - x_i)
        x_ip1 = x_i - (f / f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 1:
            ak = math.log10(
                abs(x_ip1 - x_ip2) / abs(x_ip2 - x_ip3) if abs(x_ip1 - x_ip2) / abs(x_ip2 - x_ip3) > 0 else 1)
            alpha = ak / ak2
            sequence_alpha.append(alpha if ak != 0 else 'converged')
            if x_ip1 == x_i:
                forward_error = abs(x_ip1 - 1)
                backward_error = math.log(3 - x_ip1) + x_ip1 - 2
                f_r = math.log(3 - 2) + 2 - 2
                f_prime_r = -(2 - 2) / (3 - 2)
                iter_count += 1
                return (f_r, f_prime_r, f_double, f_triple, iter_count, x_ip1, pk_list, sequence_alpha, forward_error,
                        backward_error)
        x_i = x_ip1
        iter_count += 1



def modifiedNM_b_previous(x_i):
    iter_count = 0
    pk_list = []
    sequence_alpha = []
    x_ip1 = 0
    x_ip2 = 1
    ak = 1
    for val in range(100):
        ak2 = ak
        x_ip3 = x_ip2
        x_ip2 = x_ip1
        f = math.log(3 - x_i) + x_i - 2
        f_prime = -(x_i - 2) / (3 - x_i)
        x_ip1 = x_i - (2 * f / f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 1:
            ak = math.log10(
                abs(x_ip1 - x_ip2) / abs(x_ip2 - x_ip3) if abs(x_ip1 - x_ip2) / abs(x_ip2 - x_ip3) > 0 else 1)
            alpha = ak / ak2
            sequence_alpha.append(alpha if ak != 0 else 'converged')
            if x_ip1 == x_i:
                forward_error = abs(x_ip1 - 1)
                backward_error = math.log(3 - x_ip1) + x_ip1 - 2
                iter_count += 1
                return (iter_count, x_ip1, pk_list, sequence_alpha, forward_error, backward_error)
        x_i = x_ip1
        iter_count += 1
'''

"""
The function newtonsMethod takes in the input x_i as the initial guess.
The outputs are the function, 1st derivative, 2nd derivative, 3rd derivative evaluated at the root,
number of iterations, value it converges to, list of each approximation (p_k), 
the alpha convergence sequences, forward error, and backward error. 

The function modifiedNM takes in inputs x_i as the initial guess. The outputs are the number of iterations, 
value it converges to, list of each approximation (p_k), the alpha convergence sequences, forward error, 
and backward error. 



import math


def newtonsMethod_a(x_i):
    pk_list = [x_i]             # initialize p_k list
    sequence_alpha = []         # list of alphas
    f_double = 2 * math.exp(1 - 1) - 2          # 2nd derivative eval at true root
    f_triple = 2 * math.exp(1 - 1)              # 3rd derivative eval at true root
    f_r = 2 * math.exp(1 - 1) - 1 ** 2 - 1      # eval at true root
    f_prime_r = 2 * math.exp(1 - 1) - 2 * 1     # eval at true root
    for iter_count in range(100):               # for loop through iteration
        print('Outside', iter_count, pk_list)
        f = 2 * math.exp(pk_list[iter_count] - 1) - pk_list[iter_count] ** 2 - 1        # function eval at ...
        f_prime = 2 * math.exp(pk_list[iter_count] - 1) - 2 * pk_list[iter_count]      # 1st derivative at ...
        x_ip1 = pk_list[iter_count] - (f/f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 4 and pk_list[iter_count] != x_ip1:
            #ak = math.log10(abs(x_ip1 - x_ip2)/abs(x_ip2 - x_ip3) if abs(x_ip1 - x_ip2)/abs(x_ip2 - x_ip3) > 0 else 1)
            print('Inside', iter_count, pk_list)
            ak = math.log10(
                abs(pk_list[iter_count] - pk_list[iter_count-1]) / abs(pk_list[iter_count-1] - pk_list[iter_count-2]))
            print('alpha_num', ak)
            ak2 = math.log10(
                abs(pk_list[iter_count-1] - pk_list[iter_count - 2]) / abs(pk_list[iter_count - 2] - pk_list[iter_count - 3]))
            print('alpha_denom', ak2)
            alpha = ak/ak2
            print('alpha', alpha)
            sequence_alpha.append(alpha)
            #print('alpha_sequence', sequence_alpha)
            #print('length', len(sequence_alpha))
            #print('iter_count', iter_count)
            #print('pk_list', pk_list)
        if x_ip1 == pk_list[iter_count]:
            forward_error = abs(x_ip1 - 1)
            backward_error = 2 * math.exp(x_ip1 - 1) - x_ip1 ** 2 - 1
            #print('sequence alpha', sequence_alpha)
            #print('f_r', f_r)
            return (f_r, f_prime_r, f_double, f_triple, iter_count, x_ip1, pk_list, sequence_alpha, forward_error,
                    backward_error)

a,b,c,d,e,f,g,h,i,j = newtonsMethod_a(2)


"""




def newtonsMethod_a(x_i):
    pk_list = [x_i]             # initialize p_k list
    sequence_alpha = []         # list of alphas
    f_double = 2 * math.exp(1 - 1) - 2          # 2nd derivative eval at true root
    f_triple = 2 * math.exp(1 - 1)              # 3rd derivative eval at true root
    f_r = 2 * math.exp(1 - 1) - 1 ** 2 - 1      # eval at true root
    f_prime_r = 2 * math.exp(1 - 1) - 2 * 1     # eval at true root
    for iter_count in range(100):               # for loop through iteration
        print('Outside', iter_count, pk_list)
        f = 2 * math.exp(pk_list[iter_count] - 1) - pk_list[iter_count] ** 2 - 1        # function eval at ...
        f_prime = 2 * math.exp(pk_list[iter_count] - 1) - 2 * pk_list[iter_count]      # 1st derivative at ...
        x_ip1 = pk_list[iter_count] - (f/f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 4 and pk_list[iter_count] != x_ip1:
            ak = math.log10(
                abs(pk_list[iter_count+1] - pk_list[iter_count]) / abs(pk_list[iter_count] - pk_list[iter_count-1]))      # changed to itercount+1
            ak2 = math.log10(
                abs(pk_list[iter_count] - pk_list[iter_count - 1]) / abs(pk_list[iter_count - 1] - pk_list[iter_count - 2]))      # changed to itercount+1
            alpha = ak/ak2
            sequence_alpha.append(alpha)
        if x_ip1 == pk_list[iter_count]:
            forward_error = abs(x_ip1 - 1)
            backward_error = 2 * math.exp(x_ip1 - 1) - x_ip1 ** 2 - 1
            iter_count += 1                                     #
            return (f_r, f_prime_r, f_double, f_triple, iter_count, x_ip1, pk_list, sequence_alpha, forward_error,
                    backward_error)
        iter_count += 1

def modifiedNM_a(x_i):
    pk_list = [x_i]             # initialize p_k list
    sequence_alpha = []         # list of alphas
    for iter_count in range(100):               # for loop through iteration
        f = 2 * math.exp(pk_list[iter_count] - 1) - pk_list[iter_count] ** 2 - 1        # function eval at ...
        f_prime = 2 * math.exp(pk_list[iter_count] - 1) - 2 * pk_list[iter_count]      # 1st derivative at ...
        x_ip1 = pk_list[iter_count] - (3*f/f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 4 and pk_list[iter_count] != x_ip1:          # condition to calculate true alpha values
            ak = math.log10(
                abs(pk_list[iter_count + 1] - pk_list[iter_count]) / abs(
                    pk_list[iter_count] - pk_list[iter_count - 1]))
            ak2 = math.log10(
                abs(pk_list[iter_count] - pk_list[iter_count - 1]) / abs(
                    pk_list[iter_count - 1] - pk_list[iter_count - 2]))
            alpha = ak/ak2
            sequence_alpha.append(alpha)
        if x_ip1 == pk_list[iter_count]:
            forward_error = abs(x_ip1 - 1)
            backward_error = 2 * math.exp(x_ip1 - 1) - x_ip1 ** 2 - 1
            iter_count += 1
            return (iter_count, x_ip1, pk_list, sequence_alpha, forward_error,
                    backward_error)
        iter_count += 1


def newtonsMethod_b(x_i):
    pk_list = [x_i]             # initialize p_k list
    sequence_alpha = []         # list of alphas
    f_double = -1 / (3 - 2) ** 2                # 2nd derivative eval at true root
    f_triple = 2 / (3 - 2) ** 2                 # 3rd derivative eval at true root
    f_r = math.log(3 - 2) + 2 - 2               # eval at true root
    f_prime_r = -(2 - 2) / (3 - 2)              # eval at true root
    for iter_count in range(100):               # for loop through iteration
        f = math.log(3 - pk_list[iter_count]) + pk_list[iter_count] - 2  # function eval at ...
        f_prime = -(pk_list[iter_count] - 2) / (3 - pk_list[iter_count])  # 1st derivative at ...
        x_ip1 = pk_list[iter_count] - (f/f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 4 and pk_list[iter_count] != x_ip1:
            ak = math.log10(
                abs(pk_list[iter_count + 1] - pk_list[iter_count]) / abs(
                    pk_list[iter_count] - pk_list[iter_count - 1]))
            ak2 = math.log10(
                abs(pk_list[iter_count] - pk_list[iter_count - 1]) / abs(
                    pk_list[iter_count - 1] - pk_list[iter_count - 2]))
            alpha = ak/ak2
            sequence_alpha.append(alpha)
        if x_ip1 == pk_list[iter_count]:
            forward_error = abs(x_ip1 - 2)
            backward_error = math.log(3 - pk_list[iter_count]) + pk_list[iter_count] - 2
            iter_count += 1                                     # increment
            return (f_r, f_prime_r, f_double, f_triple, iter_count, x_ip1, pk_list, sequence_alpha, forward_error,
                    backward_error)
        iter_count += 1




def modifiedNM_b(x_i):
    pk_list = [x_i]             # initialize p_k list
    sequence_alpha = []         # list of alphas
    for iter_count in range(100):               # for loop through iteration
        f = math.log(3 - pk_list[iter_count]) + pk_list[iter_count] - 2                  # function eval at ...
        f_prime = -(pk_list[iter_count] - 2) / (3 - pk_list[iter_count])                # 1st derivative at ...
        x_ip1 = pk_list[iter_count] - (2*f/f_prime)
        pk_list.append(x_ip1)
        if len(pk_list) >= 4 and pk_list[iter_count] != x_ip1:
            ak = math.log10(
                abs(pk_list[iter_count + 1] - pk_list[iter_count]) / abs(
                    pk_list[iter_count] - pk_list[iter_count - 1]))
            ak2 = math.log10(
                abs(pk_list[iter_count] - pk_list[iter_count - 1]) / abs(
                    pk_list[iter_count - 1] - pk_list[iter_count - 2]))
            alpha = ak/ak2
            sequence_alpha.append(alpha)
        if x_ip1 == pk_list[iter_count]:
            forward_error = abs(x_ip1 - 2)
            backward_error = math.log(3 - pk_list[iter_count]) + pk_list[iter_count] - 2
            iter_count += 1                                     # increment
            return (iter_count, x_ip1, pk_list, sequence_alpha, forward_error,
                    backward_error)
        iter_count += 1



# Programming problems for 1.5
def secantMethod_a(x_0,x_1):
    for selection in range(100):
        f = x_0 ** 3 - 2 * x_0 - 2
        g = x_1 ** 3 - 2 * x_0 - 2
        c = x_1 - (g*(x_1-x_0))/(g-f)
        new_c = c ** 3 - 2 * c - 2
        if new_c == 0:
            return c
        elif f*new_c < 0:
            x_1 = c
        else:
            x_0 = c
    return c



def secantMethod_c(x_0,x_1):
    for selection in range(100):
        f = math.exp(x_0) + math.sin(x_0) - 4
        g = math.exp(x_1) + math.sin(x_1) - 4
        c = x_1 - (g*(x_1-x_0))/(g-f)
        new_c = math.exp(c) + math.sin(c) - 4
        if new_c == 0:
            return c
        elif f*new_c < 0:
            x_1 = c
        else:
            x_0 = c
    return c


def methodofFalsePosition_a(a,b):
    f = a**3 - 2*a - 2
    g = b**3 - 2*b - 2
    for selection in range(100):
        c = (b*f - a*g)/(f-g)
        new_c = c**3 - 2*c - 2
        if new_c == 0:
            return c
        elif f*new_c < 0:
            b = c
        else:
            a = c
    return c


def methodofFalsePosition_c(a,b):
    f = math.exp(a) + math.sin(a) -4
    g = math.exp(b) + math.sin(b) - 4
    for selection in range(100):
        c = (b*f - a*g)/(f-g)
        new_c = math.exp(c) + math.sin(c) -4
        if new_c == 0:
            return c
        elif f*new_c < 0:
            b = c
        else:
            a = c
    return c






if __name__ == "__main__":
   print('-' * 15, 'Newtons Method A', '-' * 15)
   a, b, c, d, e, f, g, h, i, j= newtonsMethod_a(2)
   print(f'f equals {a} when evaluated at r = 1.')
   print(f'f_prime equals {b} when evaluated at r = 1.')
   print(f'f_double equals {c} when evaluated at r = 1.')
   print(f'f^(3)(x) equals {d} when evaluated at r = 1 and implies the root has multiplicity 3.')
   print(f'It took {e} iterations to converge to {f}.')
   print(f'The difference between the actual root, x = 1, and the estimation, {f}, using NM is {f-1}.')
   print(f'The list of each approximation p_k is {g}.')
   print(f'Part (A) alpha sequence {h}.')
   print(f'The forward error is {i}, and the backward error is {j}.')
   print()
   print('-' * 15, 'Modified Newtons Method A', '-' * 15)
   em, fm, gm, hm, im, jm = modifiedNM_a(2)
   print(f'It took {em} iterations using modified NM to converge to {fm}.')
   print(f'The difference between the actual root, x = 1 and the modified NM estimation, {fm}, is {fm-1}.')
   print(f'The list of each modified NM approximation p_k is {gm}.')
   print(f'Part (A) alpha sequence using modified NM is {hm}.')
   print(f'The forward error is {im}, and the backward error is {jm} using modified NM.')
   print()
   print('-' * 15, 'Newtons Method B', '-' * 15)
   ab, bb, cb, db, eb, fb, gb, hb, ib, jb = newtonsMethod_b(1)
   print(f'f equals {ab} when evaluated at r = 2.')
   print(f'f_prime equals {bb} when evaluated at r = 2.')
   print(f'f_double equals {cb} when evaluated at r = 2 and implies the root has multiplicity 2.')
   print(f'It took {eb} iterations to converge to {fb}.')
   print(f'The difference between the actual root, x = 2, and the estimation, {fb}, using NM is {fb - 2}.')
   print(f'The list of each approximation p_k is {gb}.')
   print(f'Part (B) alpha sequence {hb}.')
   print(f'The forward error is {ib}, and the backward error is {jb}.')
   print()
   print('-' * 15, 'Modified Newtons Methodr B', '-' * 15)
   emb, fmb, gmb, hmb, imb, jmb = modifiedNM_b(1)
   print(f'It took {emb} iterations using modified NM to converge to {fmb}.')
   print(f'The difference between the actual root, x = 2 and the modified NM estimation, {fmb}, is {fmb - 2}.')
   print(f'The list of each modified NM approximation p_k is {gmb}.')
   print(f'Part (B) alpha sequence using modified NM is {hmb}.')
   print(f'The forward error is {imb}, and the backward error is {jmb} using modified NM.')
   print()
   print('-' * 15, 'Secant Method A', '-' * 15)
   m = secantMethod_a(1,2)
   print(f'Applying the Secant Method to f(x) = x^3 - 2x -2 shows convergence to the root {m}.')
   print()
   print('-' * 15, 'Secant Method C', '-' * 15)
   n = secantMethod_c(1, 2)
   print(f'Applying the Secant Method to f(x) = e^x + sin(x) - 4 shows convergence to the root {n}.')
   print()
   print('-' * 15, 'Method of False Position A', '-' * 15)
   z = methodofFalsePosition_a(1, 2)
   print(f'Applying the Method of False Position to f(x) = x^3 - 2x -2 shows convergence to the root {z}.')
   print()
   print('-' * 15, 'Method of False Position C', '-' * 15)
   q = methodofFalsePosition_c(1, 2)
   print(f'Applying the Method of False Position to f(x) = e^x + sin(x) - 4 shows convergence to the root {q}.')


   plt.rcParams['pdf.fonttype'] = 42
   plt.rcParams['ps.fonttype'] = 42
   import seaborn
   seaborn.set(style='ticks')
   plt.figure(1)
   plt.rc('axes', labelsize=18)
   plt.rc('xtick', labelsize=18)
   plt.rc('ytick', labelsize=18)
   plt.rc('axes', titlesize=18)
   fig, ax = plt.subplots(1, 1, tight_layout=True)
   ax.scatter(range(len(h)), h, s=100, linewidths=2.5, facecolors='none', edgecolors='b')
   ax.scatter(range(len(hm)), hm, s=100, linewidths=2.5, facecolors='none', edgecolors='r')
   #ax.axis('equal')                     # may not want in all situations
   plt.xlabel('Number of iterations', size=16)
   plt.ylabel('Estimate of alpha', size=16)
   # plt.savefig('test.png',transparent=True)
   plt.show()

  # Figure 2
   plt.rcParams['pdf.fonttype'] = 42
   plt.rcParams['ps.fonttype'] = 42
   import seaborn
   seaborn.set(style='ticks')
   plt.figure(2)
   plt.rc('axes', labelsize=18)
   plt.rc('xtick', labelsize=18)
   plt.rc('ytick', labelsize=18)
   plt.rc('axes', titlesize=18)
   fig, ax = plt.subplots(1, 1, tight_layout=True)
   ax.scatter(range(len(hb)), hb, s=100, linewidths=2.5, facecolors='none', edgecolors='b')
   ax.scatter(range(len(hmb)), hmb, s=100, linewidths=2.5, facecolors='none', edgecolors='r')
   #ax.axis('equal')                     # may not want in all situations
   plt.xlabel('Number of iterations', size=16)
   plt.ylabel('Estimate of alpha', size=16)
   # plt.savefig('test.png',transparent=True)
   plt.show()

'''
   plt.plot(h)
   plt.plot(hm, '.')
   plt.show()

   plt.figure(2)
   plt.plot(hb)
   plt.plot(hmb, '.')
   plt.show()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
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

   plt.plot(h[0:26])
   plt.show()
'''
