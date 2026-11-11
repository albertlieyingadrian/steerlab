import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert swap_List([12, 35, 9, 56, 24]) == [24, 35, 9, 56, 12]
except Exception:
    failed += 1
try:
    assert swap_List([1, 2, 3]) == [3, 2, 1]
except Exception:
    failed += 1
try:
    assert swap_List([4, 5, 6]) == [6, 5, 4]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
