import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert big_diff([1,2,3,4]) == 3
except Exception:
    failed += 1
try:
    assert big_diff([4,5,12]) == 8
except Exception:
    failed += 1
try:
    assert big_diff([9,2,3]) == 7
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
