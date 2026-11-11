import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert find_Rotations("aaaa") == 1
except Exception:
    failed += 1
try:
    assert find_Rotations("ab") == 2
except Exception:
    failed += 1
try:
    assert find_Rotations("abc") == 3
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
