import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert min_Jumps((3,4),11)==3.5
except Exception:
    failed += 1
try:
    assert min_Jumps((3,4),0)==0
except Exception:
    failed += 1
try:
    assert min_Jumps((11,14),11)==1
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
