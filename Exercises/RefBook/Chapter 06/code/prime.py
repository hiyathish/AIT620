import itertools as it
from kanren import run, var, membero, conde, eq
from kanren.goals import success, fail
from unification import isvar
from sympy.ntheory.generate import prime, isprime

# Limit prime generation to avoid infinite search
MAX_PRIMES = 200   # adjust as needed

def check_prime(x):
    if isvar(x):
        # generate only first MAX_PRIMES primes
        return conde(*[[eq(x, prime(i))] for i in range(1, MAX_PRIMES)])
    else:
        return success if isprime(x) else fail

x = var()

list_nums = (23, 4, 27, 17, 13, 10, 21, 29, 3, 32, 11, 19)

print("\nList of primes in the list:")
print(set(run(0, x, membero(x, list_nums), check_prime(x))))

print("\nList of first 7 prime numbers:")
print(run(7, x, check_prime(x)))
