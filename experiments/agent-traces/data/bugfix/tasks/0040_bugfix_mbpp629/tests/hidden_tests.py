import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert Split([1,2,3,4,5]) == [2,4]
except Exception:
    failed += 1
try:
    assert Split([4,5,6,7,8,0,1]) == [4,6,8,0]
except Exception:
    failed += 1
try:
    assert Split ([8,12,15,19]) == [8,12]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
