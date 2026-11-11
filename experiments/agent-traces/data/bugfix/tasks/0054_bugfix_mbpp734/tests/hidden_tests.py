import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert sum_Of_Subarray_Prod([1,2,3]) == 20
except Exception:
    failed += 1
try:
    assert sum_Of_Subarray_Prod([1,2]) == 5
except Exception:
    failed += 1
try:
    assert sum_Of_Subarray_Prod([1,2,3,4]) == 84
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
