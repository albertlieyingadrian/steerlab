import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert count_reverse_pairs(["julia", "best", "tseb", "for", "ailuj"])== 2
except Exception:
    failed += 1
try:
    assert count_reverse_pairs(["geeks", "best", "for", "skeeg"]) == 1
except Exception:
    failed += 1
try:
    assert count_reverse_pairs(["makes", "best", "sekam", "for", "rof"]) == 2
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
