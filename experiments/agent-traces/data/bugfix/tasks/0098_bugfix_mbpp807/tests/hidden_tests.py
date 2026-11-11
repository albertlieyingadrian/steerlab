import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert first_odd([1,3,5]) == 1
except Exception:
    failed += 1
try:
    assert first_odd([2,4,1,3]) == 1
except Exception:
    failed += 1
try:
    assert first_odd ([8,9,1]) == 9
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
