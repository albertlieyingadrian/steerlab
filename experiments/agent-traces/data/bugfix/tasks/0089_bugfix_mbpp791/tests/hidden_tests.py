import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert remove_nested((1, 5, 7, (4, 6), 10)) == (1, 5, 7, 10)
except Exception:
    failed += 1
try:
    assert remove_nested((2, 6, 8, (5, 7), 11)) == (2, 6, 8, 11)
except Exception:
    failed += 1
try:
    assert remove_nested((3, 7, 9, (6, 8), 12)) == (3, 7, 9, 12)
except Exception:
    failed += 1
try:
    assert remove_nested((3, 7, 9, (6, 8), (5,12), 12)) == (3, 7, 9, 12)
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
