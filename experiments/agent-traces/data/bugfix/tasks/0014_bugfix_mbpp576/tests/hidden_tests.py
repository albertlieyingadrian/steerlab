import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert is_Sub_Array([1,4,3,5],[1,2]) == False
except Exception:
    failed += 1
try:
    assert is_Sub_Array([1,2,1],[1,2,1]) == True
except Exception:
    failed += 1
try:
    assert is_Sub_Array([1,0,2,2],[2,2,0]) ==False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
