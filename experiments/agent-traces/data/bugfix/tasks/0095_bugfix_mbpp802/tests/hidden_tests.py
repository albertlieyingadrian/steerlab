import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert count_rotation([3,2,1]) == 1
except Exception:
    failed += 1
try:
    assert count_rotation([4,5,1,2,3]) == 2
except Exception:
    failed += 1
try:
    assert count_rotation([7,8,9,1,2,3]) == 3
except Exception:
    failed += 1
try:
    assert count_rotation([1,2,3]) == 0
except Exception:
    failed += 1
try:
    assert count_rotation([1,3,2]) == 2
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
