import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert pair_xor_Sum([5,9,7,6],4) == 47
except Exception:
    failed += 1
try:
    assert pair_xor_Sum([7,3,5],3) == 12
except Exception:
    failed += 1
try:
    assert pair_xor_Sum([7,3],2) == 4
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
