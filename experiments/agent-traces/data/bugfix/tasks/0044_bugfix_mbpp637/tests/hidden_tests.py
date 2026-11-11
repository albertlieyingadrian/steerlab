import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert noprofit_noloss(1500,1200)==False
except Exception:
    failed += 1
try:
    assert noprofit_noloss(100,100)==True
except Exception:
    failed += 1
try:
    assert noprofit_noloss(2000,5000)==False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
