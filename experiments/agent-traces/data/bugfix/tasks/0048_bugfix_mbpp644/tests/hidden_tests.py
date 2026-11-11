import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert reverse_Array_Upto_K([1, 2, 3, 4, 5, 6],4) == [4, 3, 2, 1, 5, 6]
except Exception:
    failed += 1
try:
    assert reverse_Array_Upto_K([4, 5, 6, 7], 2) == [5, 4, 6, 7]
except Exception:
    failed += 1
try:
    assert reverse_Array_Upto_K([9, 8, 7, 6, 5],3) == [7, 8, 9, 6, 5]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
