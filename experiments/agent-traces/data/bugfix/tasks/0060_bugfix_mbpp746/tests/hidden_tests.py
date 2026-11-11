import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert sector_area(4,45)==6.283185307179586
except Exception:
    failed += 1
try:
    assert sector_area(9,45)==31.808625617596654
except Exception:
    failed += 1
try:
    assert sector_area(9,361)==None
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
