import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert _sum([1, 2, 3]) == 6
except Exception:
    failed += 1
try:
    assert _sum([15, 12, 13, 10]) == 50
except Exception:
    failed += 1
try:
    assert _sum([0, 1, 2]) == 3
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
