import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert check_K((10, 4, 5, 6, 8), 6) == True
except Exception:
    failed += 1
try:
    assert check_K((1, 2, 3, 4, 5, 6), 7) == False
except Exception:
    failed += 1
try:
    assert check_K((7, 8, 9, 44, 11, 12), 11) == True
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
