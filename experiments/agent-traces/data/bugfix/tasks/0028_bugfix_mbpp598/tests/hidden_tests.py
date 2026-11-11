import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert armstrong_number(153)==True
except Exception:
    failed += 1
try:
    assert armstrong_number(259)==False
except Exception:
    failed += 1
try:
    assert armstrong_number(4458)==False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
