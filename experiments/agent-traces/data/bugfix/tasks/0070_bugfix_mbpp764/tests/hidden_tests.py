import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert number_ctr('program2bedone') == 1
except Exception:
    failed += 1
try:
    assert number_ctr('3wonders') == 1
except Exception:
    failed += 1
try:
    assert number_ctr('123') == 3
except Exception:
    failed += 1
try:
    assert number_ctr('3wond-1ers2') == 3
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
