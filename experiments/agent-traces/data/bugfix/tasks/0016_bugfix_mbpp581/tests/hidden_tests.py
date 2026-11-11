import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert surface_Area(3,4) == 33
except Exception:
    failed += 1
try:
    assert surface_Area(4,5) == 56
except Exception:
    failed += 1
try:
    assert surface_Area(1,2) == 5
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
