import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert sum_in_range(2,5) == 8
except Exception:
    failed += 1
try:
    assert sum_in_range(5,7) == 12
except Exception:
    failed += 1
try:
    assert sum_in_range(7,13) == 40
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
