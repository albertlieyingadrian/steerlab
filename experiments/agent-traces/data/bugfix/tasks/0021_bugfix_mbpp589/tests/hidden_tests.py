import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert perfect_squares(1,30)==[1, 4, 9, 16, 25]
except Exception:
    failed += 1
try:
    assert perfect_squares(50,100)==[64, 81, 100]
except Exception:
    failed += 1
try:
    assert perfect_squares(100,200)==[100, 121, 144, 169, 196]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
