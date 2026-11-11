import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert divisible_by_digits(1,22)==[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
except Exception:
    failed += 1
try:
    assert divisible_by_digits(1,15)==[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15]
except Exception:
    failed += 1
try:
    assert divisible_by_digits(20,25)==[22, 24]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
