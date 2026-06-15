from core.gps_utils import *

# Entry point kept for optional command-line tests.
# The V3 application is launched from ui/streamlit_app.py.
# ./scripts/run_mac.sh

p1 = (50.104493, 14.394761)
p2 = (50.105000, 14.395500)

print(haversine_m(p1, p2))
print(calculate_heading(p1, p2))
print(interpolate_points(p1, p2, 5))
