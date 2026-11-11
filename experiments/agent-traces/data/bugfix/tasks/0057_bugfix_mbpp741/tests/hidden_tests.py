import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert all_Characters_Same("python") == False
except Exception:
    failed += 1
try:
    assert all_Characters_Same("aaa") == True
except Exception:
    failed += 1
try:
    assert all_Characters_Same("data") == False
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
