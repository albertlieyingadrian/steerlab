import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert multiply_elements((1, 5, 7, 8, 10)) == (5, 35, 56, 80)
except Exception:
    failed += 1
try:
    assert multiply_elements((2, 4, 5, 6, 7)) == (8, 20, 30, 42)
except Exception:
    failed += 1
try:
    assert multiply_elements((12, 13, 14, 9, 15)) == (156, 182, 126, 135)
except Exception:
    failed += 1
try:
    assert multiply_elements((12,)) == ()
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
