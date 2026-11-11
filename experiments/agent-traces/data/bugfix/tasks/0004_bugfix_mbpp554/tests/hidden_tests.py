import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert Split([1,2,3,4,5,6]) == [1,3,5]
except Exception:
    failed += 1
try:
    assert Split([10,11,12,13]) == [11,13]
except Exception:
    failed += 1
try:
    assert Split([7,8,9,1]) == [7,9,1]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
