import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert check_min_heap([1, 2, 3, 4, 5, 6]) == True
except Exception:
    failed += 1
try:
    assert check_min_heap([2, 3, 4, 5, 10, 15]) == True
except Exception:
    failed += 1
try:
    assert check_min_heap([2, 10, 4, 5, 3, 15]) == False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
