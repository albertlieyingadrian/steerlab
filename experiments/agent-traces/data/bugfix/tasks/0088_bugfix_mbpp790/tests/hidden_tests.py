import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert even_position([3,2,1]) == False
except Exception:
    failed += 1
try:
    assert even_position([1,2,3]) == False
except Exception:
    failed += 1
try:
    assert even_position([2,1,4]) == True
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
