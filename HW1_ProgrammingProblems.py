# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
 #   print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# Computer Problems

## Question 1

### a) Declare two variables, x and epsilon. Initialize epsilon = 1.
        # Create a loop, and each time through the loop, assign x = 1.0 + epsilon and then divide epsilon by 2.
        # Repeat until the value of x - 1.0 is no longer positive.



import math

x = 10
count = 0
epsilon = float(1.0)
while x - 1.0 > 0.0:
    epsilon = epsilon / 2.0
    x = 1.0 + epsilon
    print('new x', x)
    print('new epsilon', epsilon)
    count += 1
print(x)
print(epsilon)
print(count-1)



### b) Use the result from (a) to determine the number of bits in the mantissa for the program language.
print('There are 54 bits in the mantissa.')
# convert to binary
## in binary: 000000000000000000000000000000000000000000000000000001
# mantissa sig to the right
# 1.1102230246251565e-16  --> potentially this is the correct one
x = 0.00000000000000005551115123125783
list = []
while x != 1.0:
    x *= 2
    print('#1b x', x)
print('final #1b', x)

#000000000000000000000000000000000000000000000000000001




x = 0.00000000000000005551115123125783 == 5.551115123125783e-17
print(x)
count = 0
for digit in '0.00000000000000005551115123125783':
    print(digit)
    count += 1
print(count)

c = 0.00000000000000005551115123125783
count = 0
while c != 1.00:
    c *= 2
    print('new epsilon', c)
    count += 1
print('final epsilon', c)
print(count)


## 2)
###a) Initialize a variable r=1.0.
# Create a loop that prints out a value for r and for log2r and then doubles r each time
# through the loop. Repeat until the value of r becomes infinite. Find the value k for the largest 2k before overflow.

r = 1.0
k = 0
while math.isinf(r) != True:
    print(r)
    print(math.log2(r))
    r = r*2
    k += 1
print('The value of k for the largest 2^k before overflow is', k-1)  # Equivalent to math.log2(r).


q = 8.98846567431158e+307
count = 0
while q//2 != 0.0:
    r = q % 2
    q = q//2
    print('mantissa', q)
    print('Remainder', r)
    count += 1
print('final for mantissa', q)
print('final remainder', r)
print(count)



### b) Use the result in (a) to determine the total number of bits in the exponent of the program languages data type.
### Based on the previous question, determine the total number of bits in each data type.
## may need to convert to binary. 2^c-1023, and 1+mantissa
# c is at most 2046 --> convert to binary
# data types are the c and mantissa
# total, add up mantissa, exponent, and sign
#8.98846567431158e+307
print('The total number of bits in the exponent is 11 because (2046)_10 = (11111111110)_2. Thus '
      'it has 11 bits.')
print('The sign has', 1, 'bit and is', 1, 'because the number is positive')
print('The exponent, 2046, has 11 bits')
print('The mantissa has', 52, 'bits because for double precision the bits are divided among the parts as such.')

c = 2046
count = 0
while c//2 != 0.0:
    r = c % 2
    c = c//2
    print('Exponent', c)
    print('Remainder', r)
    count += 1
print('final exponent', c)
print('final remainder', r)
print(count)

# The sign has 1 bit.
# With c at most 2046, (2046)_10 = (11111111110)_2, which means there are 11 bits for the exponent.
# The mantissa has

#x = 32
#epsilon = float(1.0)

#while x - 1.0 > 0.0:
 #   x = 1.0 + epsilon
 #   epsilon = epsilon/2.0#
#print(x)
#print(epsilon)
#print(type(epsilon))
#float2int = math.frexp(epsilon)[1]-1
#print('There are',float2int, 'number of bits in the mantissa')
#print(math.log(10,2))
#print(float(10))

x = ((4/3) - (1/3)) - 1
print('#9', x)



x = 2**-4
y = 2**-2
print('2^-4', x, 'y = ', y)
#1.1102230246251565e-16


x = -0.33 * 2**-52
print('-0.33 * 2^-52', x)


#x = ((1.0010101010101010101010101010101010101010101010101011
#    - 1.0101010101010101010101010101010101010101010101010101

 #1.0010101010101010101010101010101010101010101010101011
#+0.1010101010101010101010101010101010101010101010101011
#+0.1111111111111111111111111111111111111111111111111110)

x = 2**10 + 2**9 +2**8 +2**7 + 2**6 + 2**5 + 2**4 + 2**3 + 2**2 + 2**1
print(x)

print('9a', x)

#2.220446049250313e-16
x = 2046
while x // 2 != 0.0:
    y = x % 2
    x //= 2
    print('integer',x)
    print('remainder', y)

  #  001100000000000000000000000000000000000000000000000
   # 1001100110011001100110011001100110011001100110011010

n = 1
total = 0
while n <= 54:
    z = 2**-n
    print('this is ', z)
    n += 1
    print('this is', n)
    total += z
    print(total)
print('The total is', total, 'and the exponent used is', n-1)

x = 1-total
print(x)


"""  
# Python program to convert float
# decimal to binary number

# Function returns octal representation
def float_bin(number, places = 3):

	# split() separates whole number and decimal
	# part and stores it in two separate variables
	whole, dec = str(number).split(".")

	# Convert both whole number and decimal
	# part from string type to integer type
	whole = int(whole)
	dec = int (dec)

	# Convert the whole number part to it's
	# respective binary form and remove the
	# "0b" from it.
	res = bin(whole).lstrip("0b") + "."

	# Iterate the number of times, we want
	# the number of decimal places to be
	for x in range(places):

		# Multiply the decimal value by 2
		# and separate the whole number part
		# and decimal part
		whole, dec = str((decimal_converter(dec)) * 2).split(".")

		# Convert the decimal part
		# to integer again
		dec = int(dec)

		# Keep adding the integer parts
		# receive to the result variable
		res += whole

	return res

# Function converts the value passed as
# parameter to it's decimal representation
def decimal_converter(num):
	while num > 1:
		num /= 10
	return num

# Driver Code

# Take the user input for
# the floating point number
n = input("Enter your floating point value : \n")

# Take user input for the number of
# decimal places user want result as
p = int(input("Enter the number of decimal places of the result : \n"))

#print(float_bin(n, places = p))
"""
