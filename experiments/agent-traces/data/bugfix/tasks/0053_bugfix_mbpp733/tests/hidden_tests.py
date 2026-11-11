import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert find_first_occurrence([2, 5, 5, 5, 6, 6, 8, 9, 9, 9], 5) == 1
except Exception:
    failed += 1
try:
    assert find_first_occurrence([2, 3, 5, 5, 6, 6, 8, 9, 9, 9], 5) == 2
except Exception:
    failed += 1
try:
    assert find_first_occurrence([2, 4, 1, 5, 6, 6, 8, 9, 9, 9], 6) == 4
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
