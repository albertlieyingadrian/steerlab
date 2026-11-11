import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert split_Arr([12,10,5,6,52,36],2) == [5,6,52,36,12,10]
except Exception:
    failed += 1
try:
    assert split_Arr([1,2,3,4],1) == [2,3,4,1]
except Exception:
    failed += 1
try:
    assert split_Arr([0,1,2,3,4,5,6,7],3) == [3,4,5,6,7,0,1,2]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
