import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert find_Index(2) == 4
except Exception:
    failed += 1
try:
    assert find_Index(3) == 14
except Exception:
    failed += 1
try:
    assert find_Index(4) == 45
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
