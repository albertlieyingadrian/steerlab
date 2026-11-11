import sys
sys.path.insert(0, '/home/user/input')

from solution import *
failed = 0
try:
    assert polar_rect(3,4)==((5.0, 0.9272952180016122), (-2+2.4492935982947064e-16j))
except Exception:
    failed += 1
try:
    assert polar_rect(4,7)==((8.06225774829855, 1.0516502125483738), (-2+2.4492935982947064e-16j))
except Exception:
    failed += 1
try:
    assert polar_rect(15,17)==((22.67156809750927, 0.8478169733934057), (-2+2.4492935982947064e-16j))
except Exception:
    failed += 1
sys.exit(1 if failed else 0)
