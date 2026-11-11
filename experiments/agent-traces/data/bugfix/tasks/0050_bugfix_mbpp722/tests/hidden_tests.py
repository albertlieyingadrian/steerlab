import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert filter_data({'Cierra Vega': (6.2, 70), 'Alden Cantrell': (5.9, 65), 'Kierra Gentry': (6.0, 68), 'Pierre Cox': (5.8, 66)},6.0,70)=={'Cierra Vega': (6.2, 70)}
except Exception:
    failed += 1
try:
    assert filter_data({'Cierra Vega': (6.2, 70), 'Alden Cantrell': (5.9, 65), 'Kierra Gentry': (6.0, 68), 'Pierre Cox': (5.8, 66)},5.9,67)=={'Cierra Vega': (6.2, 70),'Kierra Gentry': (6.0, 68)}
except Exception:
    failed += 1
try:
    assert filter_data({'Cierra Vega': (6.2, 70), 'Alden Cantrell': (5.9, 65), 'Kierra Gentry': (6.0, 68), 'Pierre Cox': (5.8, 66)},5.7,64)=={'Cierra Vega': (6.2, 70),'Alden Cantrell': (5.9, 65),'Kierra Gentry': (6.0, 68),'Pierre Cox': (5.8, 66)}
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
