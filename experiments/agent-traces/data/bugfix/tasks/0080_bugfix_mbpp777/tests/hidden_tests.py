import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert find_sum([1,2,3,1,1,4,5,6]) == 21
except Exception:
    failed += 1
try:
    assert find_sum([1,10,9,4,2,10,10,45,4]) == 71
except Exception:
    failed += 1
try:
    assert find_sum([12,10,9,45,2,10,10,45,10]) == 78
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
