import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert bell_Number(2) == 2
except Exception:
    failed += 1
try:
    assert bell_Number(3) == 5
except Exception:
    failed += 1
try:
    assert bell_Number(4) == 15
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
