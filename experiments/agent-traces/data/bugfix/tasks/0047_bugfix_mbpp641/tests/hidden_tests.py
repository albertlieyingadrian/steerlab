import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert is_nonagonal(10) == 325
except Exception:
    failed += 1
try:
    assert is_nonagonal(15) == 750
except Exception:
    failed += 1
try:
    assert is_nonagonal(18) == 1089
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
