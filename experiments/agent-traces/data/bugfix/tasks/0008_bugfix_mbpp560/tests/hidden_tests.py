import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert union_elements((3, 4, 5, 6),(5, 7, 4, 10) ) == (3, 4, 5, 6, 7, 10)
except Exception:
    failed += 1
try:
    assert union_elements((1, 2, 3, 4),(3, 4, 5, 6) ) == (1, 2, 3, 4, 5, 6)
except Exception:
    failed += 1
try:
    assert union_elements((11, 12, 13, 14),(13, 15, 16, 17) ) == (11, 12, 13, 14, 15, 16, 17)
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
