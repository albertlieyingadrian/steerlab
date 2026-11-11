import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert max_sub_array_sum([-2, -3, 4, -1, -2, 1, 5, -3], 8) == 7
except Exception:
    failed += 1
try:
    assert max_sub_array_sum([-3, -4, 5, -2, -3, 2, 6, -4], 8) == 8
except Exception:
    failed += 1
try:
    assert max_sub_array_sum([-4, -5, 6, -3, -4, 3, 7, -5], 8) == 10
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
