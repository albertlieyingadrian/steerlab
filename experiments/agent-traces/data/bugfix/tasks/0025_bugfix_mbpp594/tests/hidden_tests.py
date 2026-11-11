import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert diff_even_odd([1,3,5,7,4,1,6,8])==3
except Exception:
    failed += 1
try:
    assert diff_even_odd([1,2,3,4,5,6,7,8,9,10])==1
except Exception:
    failed += 1
try:
    assert diff_even_odd([1,5,7,9,10])==9
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
