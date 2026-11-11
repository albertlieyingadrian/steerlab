import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert test_three_equal(1,1,1) == 3
except Exception:
    failed += 1
try:
    assert test_three_equal(-1,-2,-3) == 0
except Exception:
    failed += 1
try:
    assert test_three_equal(1,2,2) == 2
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
