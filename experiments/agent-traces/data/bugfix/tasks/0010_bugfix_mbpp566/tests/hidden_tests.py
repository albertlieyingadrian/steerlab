import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert sum_digits(345)==12
except Exception:
    failed += 1
try:
    assert sum_digits(12)==3
except Exception:
    failed += 1
try:
    assert sum_digits(97)==16
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
