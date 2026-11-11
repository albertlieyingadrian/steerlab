import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert find_First_Missing([0,1,2,3]) == 4
except Exception:
    failed += 1
try:
    assert find_First_Missing([0,1,2,6,9]) == 3
except Exception:
    failed += 1
try:
    assert find_First_Missing([2,3,5,8,9]) == 0
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
