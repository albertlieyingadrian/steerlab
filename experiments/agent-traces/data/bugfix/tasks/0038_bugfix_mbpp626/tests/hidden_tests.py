import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert triangle_area(-1) == None
except Exception:
    failed += 1
try:
    assert triangle_area(0) == 0
except Exception:
    failed += 1
try:
    assert triangle_area(2) == 4
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
