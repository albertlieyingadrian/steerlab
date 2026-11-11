import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert count_vowels('bestinstareels') == 7
except Exception:
    failed += 1
try:
    assert count_vowels('partofthejourneyistheend') == 12
except Exception:
    failed += 1
try:
    assert count_vowels('amazonprime') == 5
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
