import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert check_monthnumber_number(6)==True
except Exception:
    failed += 1
try:
    assert check_monthnumber_number(2)==False
except Exception:
    failed += 1
try:
    assert check_monthnumber_number(12)==False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
