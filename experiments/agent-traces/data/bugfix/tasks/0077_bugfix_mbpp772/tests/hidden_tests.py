import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert remove_length('The person is most value tet', 3) == 'person is most value'
except Exception:
    failed += 1
try:
    assert remove_length('If you told me about this ok', 4) == 'If you me about ok'
except Exception:
    failed += 1
try:
    assert remove_length('Forces of darkeness is come into the play', 4) == 'Forces of darkeness is the'
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
