import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert get_ludic(10) == [1, 2, 3, 5, 7]
except Exception:
    failed += 1
try:
    assert get_ludic(25) == [1, 2, 3, 5, 7, 11, 13, 17, 23, 25]
except Exception:
    failed += 1
try:
    assert get_ludic(45) == [1, 2, 3, 5, 7, 11, 13, 17, 23, 25, 29, 37, 41, 43]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
