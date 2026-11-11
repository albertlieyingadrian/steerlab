import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert replace_spaces('Jumanji The Jungle') == 'Jumanji_The_Jungle'
except Exception:
    failed += 1
try:
    assert replace_spaces('The_Avengers') == 'The Avengers'
except Exception:
    failed += 1
try:
    assert replace_spaces('Fast and Furious') == 'Fast_and_Furious'
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
