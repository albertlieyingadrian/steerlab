import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert two_unique_nums([1,2,3,2,3,4,5]) == [1, 4, 5]
except Exception:
    failed += 1
try:
    assert two_unique_nums([1,2,3,2,4,5]) == [1, 3, 4, 5]
except Exception:
    failed += 1
try:
    assert two_unique_nums([1,2,3,4,5]) == [1, 2, 3, 4, 5]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
