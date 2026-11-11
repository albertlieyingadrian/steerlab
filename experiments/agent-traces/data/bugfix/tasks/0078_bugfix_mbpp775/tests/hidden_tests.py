import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert odd_position([2,1,4,3,6,7,6,3]) == True
except Exception:
    failed += 1
try:
    assert odd_position([4,1,2]) == True
except Exception:
    failed += 1
try:
    assert odd_position([1,2,3]) == False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
