import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert last([1,2,3],1) == 0
except Exception:
    failed += 1
try:
    assert last([1,1,1,2,3,4],1) == 2
except Exception:
    failed += 1
try:
    assert last([2,3,2,3,6,8,9],3) == 3
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
