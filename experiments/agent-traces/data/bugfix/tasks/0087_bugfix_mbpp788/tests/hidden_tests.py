import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert new_tuple(["WEB", "is"], "best") == ('WEB', 'is', 'best')
except Exception:
    failed += 1
try:
    assert new_tuple(["We", "are"], "Developers") == ('We', 'are', 'Developers')
except Exception:
    failed += 1
try:
    assert new_tuple(["Part", "is"], "Wrong") == ('Part', 'is', 'Wrong')
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
