import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert mul_even_odd([1,3,5,7,4,1,6,8])==4
except Exception:
    failed += 1
try:
    assert mul_even_odd([1,2,3,4,5,6,7,8,9,10])==2
except Exception:
    failed += 1
try:
    assert mul_even_odd([1,5,7,9,10])==10
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
