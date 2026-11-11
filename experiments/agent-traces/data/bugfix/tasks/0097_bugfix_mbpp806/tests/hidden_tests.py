import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert max_run_uppercase('GeMKSForGERksISBESt') == 5
except Exception:
    failed += 1
try:
    assert max_run_uppercase('PrECIOusMOVemENTSYT') == 6
except Exception:
    failed += 1
try:
    assert max_run_uppercase('GooGLEFluTTER') == 4
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
