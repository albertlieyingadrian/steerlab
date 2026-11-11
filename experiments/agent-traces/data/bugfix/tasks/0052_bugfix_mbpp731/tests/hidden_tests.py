import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert lateralsurface_cone(5,12)==204.20352248333654
except Exception:
    failed += 1
try:
    assert lateralsurface_cone(10,15)==566.3586699569488
except Exception:
    failed += 1
try:
    assert lateralsurface_cone(19,17)==1521.8090132193388
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
