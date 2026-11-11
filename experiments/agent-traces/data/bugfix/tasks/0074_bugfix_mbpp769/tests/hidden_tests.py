import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert (Diff([10, 15, 20, 25, 30, 35, 40], [25, 40, 35])) == [10, 20, 30, 15]
except Exception:
    failed += 1
try:
    assert (Diff([1,2,3,4,5], [6,7,1])) == [2,3,4,5,6,7]
except Exception:
    failed += 1
try:
    assert (Diff([1,2,3], [6,7,1])) == [2,3,6,7]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
