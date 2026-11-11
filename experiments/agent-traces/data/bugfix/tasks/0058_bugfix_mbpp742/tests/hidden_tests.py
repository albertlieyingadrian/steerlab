import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert area_tetrahedron(3)==15.588457268119894
except Exception:
    failed += 1
try:
    assert area_tetrahedron(20)==692.8203230275509
except Exception:
    failed += 1
try:
    assert area_tetrahedron(10)==173.20508075688772
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
