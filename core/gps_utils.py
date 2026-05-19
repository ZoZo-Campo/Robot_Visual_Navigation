import math


# =========================================================
# DISTANCE GPS
# =========================================================

def haversine_m(point1, point2):
    """
    Compute distance between two GPS points in meters.
    """

    lat1, lon1 = point1
    lat2, lon2 = point2

    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# =========================================================
# GPS HEADING
# =========================================================

def calculate_heading(point1, point2):
    """
    Compute heading angle between two GPS points.
    Returns heading in degrees.
    """

    lat1, lon1 = point1
    lat2, lon2 = point2

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    delta_lon = math.radians(
        lon2 - lon1
    )

    x = math.sin(delta_lon) * math.cos(lat2)

    y = (
        math.cos(lat1) * math.sin(lat2)
        -
        math.sin(lat1)
        * math.cos(lat2)
        * math.cos(delta_lon)
    )

    heading = math.degrees(
        math.atan2(x, y)
    )

    return (heading + 360) % 360


# =========================================================
# GPS INTERPOLATION
# =========================================================

def interpolate_points(
    point1,
    point2,
    step_m
):
    """
    Interpolate GPS points every step_m meters.
    """

    distance = haversine_m(
        point1,
        point2
    )

    if distance < step_m:
        return [point1, point2]

    num_steps = int(distance // step_m)

    lat1, lon1 = point1
    lat2, lon2 = point2

    points = []

    for i in range(num_steps + 1):

        t = i / num_steps

        lat = lat1 + (
            lat2 - lat1
        ) * t

        lon = lon1 + (
            lon2 - lon1
        ) * t

        points.append(
            (lat, lon)
        )

    return points


# =========================================================
# POINT COMPARISON
# =========================================================

def same_point(
    point1,
    point2,
    tolerance=1e-7
):
    """
    Check if two GPS points are identical.
    """

    if point1 is None or point2 is None:
        return False

    return (
        abs(point1[0] - point2[0]) < tolerance
        and
        abs(point1[1] - point2[1]) < tolerance
    )