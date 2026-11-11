import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert jacobsthal_num(5) == 11
except Exception:
    failed += 1
try:
    assert jacobsthal_num(2) == 1
except Exception:
    failed += 1
try:
    assert jacobsthal_num(4) == 5
except Exception:
    failed += 1
try:
    assert jacobsthal_num(13) == 2731
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
