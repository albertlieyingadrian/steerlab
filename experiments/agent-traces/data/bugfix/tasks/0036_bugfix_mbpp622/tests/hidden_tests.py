import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert get_median([1, 12, 15, 26, 38], [2, 13, 17, 30, 45], 5) == 16.0
except Exception:
    failed += 1
try:
    assert get_median([2, 4, 8, 9], [7, 13, 19, 28], 4) == 8.5
except Exception:
    failed += 1
try:
    assert get_median([3, 6, 14, 23, 36, 42], [2, 18, 27, 39, 49, 55], 6) == 25.0
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
