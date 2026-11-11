import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert last_Digit_Factorial(4) == 4
except Exception:
    failed += 1
try:
    assert last_Digit_Factorial(21) == 0
except Exception:
    failed += 1
try:
    assert last_Digit_Factorial(30) == 0
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
