import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert get_coordinates((3, 4)) == [[2, 3], [2, 4], [2, 5], [3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5]]
except Exception:
    failed += 1
try:
    assert get_coordinates((4, 5)) ==[[3, 4], [3, 5], [3, 6], [4, 4], [4, 5], [4, 6], [5, 4], [5, 5], [5, 6]]
except Exception:
    failed += 1
try:
    assert get_coordinates((5, 6)) == [[4, 5], [4, 6], [4, 7], [5, 5], [5, 6], [5, 7], [6, 5], [6, 6], [6, 7]]
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
