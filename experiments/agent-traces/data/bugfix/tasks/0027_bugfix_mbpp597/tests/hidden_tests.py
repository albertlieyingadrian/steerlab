import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert find_kth([2, 3, 6, 7, 9], [1, 4, 8, 10], 5) == 6
except Exception:
    failed += 1
try:
    assert find_kth([100, 112, 256, 349, 770], [72, 86, 113, 119, 265, 445, 892], 7) == 256
except Exception:
    failed += 1
try:
    assert find_kth([3, 4, 7, 8, 10], [2, 5, 9, 11], 6) == 8
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
