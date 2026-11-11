import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert get_pairs_count([1,1,1,1],2) == 6
except Exception:
    failed += 1
try:
    assert get_pairs_count([1,5,7,-1,5],6) == 3
except Exception:
    failed += 1
try:
    assert get_pairs_count([1,-2,3],1) == 1
except Exception:
    failed += 1
try:
    assert get_pairs_count([-1,-2,3],-3) == 1
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
