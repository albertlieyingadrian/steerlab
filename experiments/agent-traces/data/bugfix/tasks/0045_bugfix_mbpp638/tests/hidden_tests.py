import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert wind_chill(120,35)==40
except Exception:
    failed += 1
try:
    assert wind_chill(40,20)==19
except Exception:
    failed += 1
try:
    assert wind_chill(10,8)==6
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
