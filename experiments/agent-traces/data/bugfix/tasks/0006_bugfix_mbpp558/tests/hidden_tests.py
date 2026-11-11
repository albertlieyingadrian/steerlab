import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert digit_distance_nums(1,2) == 1
except Exception:
    failed += 1
try:
    assert digit_distance_nums(23,56) == 6
except Exception:
    failed += 1
try:
    assert digit_distance_nums(123,256) == 7
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
