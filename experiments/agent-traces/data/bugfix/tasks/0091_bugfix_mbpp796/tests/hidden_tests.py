import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert return_sum({'a': 100, 'b':200, 'c':300}) == 600
except Exception:
    failed += 1
try:
    assert return_sum({'a': 25, 'b':18, 'c':45}) == 88
except Exception:
    failed += 1
try:
    assert return_sum({'a': 36, 'b':39, 'c':49}) == 124
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
