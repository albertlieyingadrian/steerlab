import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert swap_List([1,2,3]) == [3,2,1]
except Exception:
    failed += 1
try:
    assert swap_List([1,2,3,4,4]) == [4,2,3,4,1]
except Exception:
    failed += 1
try:
    assert swap_List([4,5,6]) == [6,5,4]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
