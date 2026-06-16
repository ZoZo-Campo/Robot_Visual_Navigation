import os
import hashlib
from pathlib import Path

from config import (
    ROUTE_CACHE_DIR,
    ROUTE_DOWNLOAD_TIMEOUT_S,
    ROUTE_MAX_RADIUS_M,
    ROUTE_USE_FALLBACK_DIRECT,
)
from core.gps_utils import (
    interpolate_points,
    haversine_m,
)


class RoutePlanner:

    def __init__(
        self,
        points,
        network_type="walk",
        step_m=5,
        padding_m=150,
        cache_dir=ROUTE_CACHE_DIR,
        max_radius_m=ROUTE_MAX_RADIUS_M,
        use_fallback_direct=ROUTE_USE_FALLBACK_DIRECT,
    ):
        self.points = points
        self.network_type = network_type
        self.step_m = step_m
        self.padding_m = padding_m
        self.cache_dir = Path(cache_dir)
        self.max_radius_m = max_radius_m
        self.use_fallback_direct = use_fallback_direct
        self.used_fallback = False
        self.cache_hit = False
        self.last_message = ""

    # =====================================================
    # GRAPH DOWNLOAD AREA
    # =====================================================

    def get_graph_center_and_radius(self):
        lats = [point[0] for point in self.points]
        lons = [point[1] for point in self.points]

        center_point = (
            (min(lats) + max(lats)) / 2,
            (min(lons) + max(lons)) / 2,
        )

        max_distance = 0

        for point in self.points:
            distance = haversine_m(
                center_point,
                point
            )

            if distance > max_distance:
                max_distance = distance

        download_radius = int(
            max_distance + self.padding_m
        )

        download_radius = max(
            download_radius,
            200
        )
        download_radius = min(
            download_radius,
            int(self.max_radius_m)
        )

        return center_point, download_radius

    def graph_cache_path(
        self,
        center_point,
        download_radius,
    ):
        rounded_center = (
            round(center_point[0], 4),
            round(center_point[1], 4),
        )
        rounded_radius = int(download_radius / 100) * 100
        key = (
            f"{rounded_center[0]}_"
            f"{rounded_center[1]}_"
            f"{rounded_radius}_"
            f"{self.network_type}"
        )
        digest = hashlib.sha1(
            key.encode("utf-8")
        ).hexdigest()[:12]
        return self.cache_dir / f"osm_{digest}.graphml"

    # =====================================================
    # BUILD GRAPH
    # =====================================================

    def build_graph(self):
        import osmnx as ox

        center_point, download_radius = (
            self.get_graph_center_and_radius()
        )

        print(f"OSM center: {center_point}")
        print(f"OSM radius: {download_radius} m")

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        cache_path = self.graph_cache_path(
            center_point,
            download_radius,
        )

        if cache_path.exists():
            self.cache_hit = True
            self.last_message = f"Loaded OSM graph cache: {cache_path.name}"
            return ox.load_graphml(cache_path)

        self.cache_hit = False
        old_timeout = getattr(
            ox.settings,
            "requests_timeout",
            None,
        )

        if old_timeout is not None:
            ox.settings.requests_timeout = ROUTE_DOWNLOAD_TIMEOUT_S

        try:
            graph = ox.graph_from_point(
                center_point,
                dist=download_radius,
                network_type=self.network_type,
                simplify=True,
            )
            ox.save_graphml(
                graph,
                filepath=cache_path,
            )
            self.last_message = f"Downloaded and cached OSM graph: {cache_path.name}"
        finally:
            if old_timeout is not None:
                ox.settings.requests_timeout = old_timeout

        return graph

    # =====================================================
    # CLOSEST NODE
    # =====================================================

    @staticmethod
    def nearest_node(graph, point):
        import osmnx as ox

        return ox.distance.nearest_nodes(
            graph,
            X=point[1],
            Y=point[0],
        )

    # =====================================================
    # INTERPOLATE RAW ROUTE
    # =====================================================

    def densify_raw_route(self, raw_route):
        dense_route = []

        for i in range(len(raw_route) - 1):
            segment = interpolate_points(
                raw_route[i],
                raw_route[i + 1],
                self.step_m,
            )

            dense_route.extend(
                segment[:-1]
            )

        dense_route.append(
            raw_route[-1]
        )

        return dense_route

    # =====================================================
    # COMPUTE SINGLE SEGMENT
    # =====================================================

    def compute_segment(
        self,
        graph,
        start_point,
        end_point,
    ):
        import networkx as nx

        start_node = self.nearest_node(
            graph,
            start_point
        )

        end_node = self.nearest_node(
            graph,
            end_point
        )

        path = nx.shortest_path(
            graph,
            start_node,
            end_node,
            weight="length",
        )

        raw_route = []

        for node in path:
            raw_route.append(
                (
                    graph.nodes[node]["y"],
                    graph.nodes[node]["x"],
                )
            )

        dense_route = self.densify_raw_route(
            raw_route
        )

        dense_route[0] = start_point
        dense_route[-1] = end_point

        return dense_route

    # =====================================================
    # COMPUTE FULL ROUTE
    # =====================================================

    def compute_route(self):
        if len(self.points) < 2:
            raise ValueError(
                "At least two points are required."
            )

        self.used_fallback = False

        try:
            graph = self.build_graph()
        except Exception as exc:
            if not self.use_fallback_direct:
                raise
            self.used_fallback = True
            self.last_message = (
                "OSM route failed; using direct interpolated GPS route. "
                f"Reason: {exc}"
            )
            return self.compute_direct_route()

        full_route = []

        for i in range(len(self.points) - 1):
            try:
                segment = self.compute_segment(
                    graph=graph,
                    start_point=self.points[i],
                    end_point=self.points[i + 1],
                )
            except Exception as exc:
                if not self.use_fallback_direct:
                    raise
                self.used_fallback = True
                self.last_message = (
                    "OSM path search failed; using direct interpolated GPS route. "
                    f"Reason: {exc}"
                )
                return self.compute_direct_route()

            if i > 0:
                segment = segment[1:]

            full_route.extend(segment)

        return full_route

    def compute_direct_route(self):
        full_route = []

        for i in range(len(self.points) - 1):
            segment = interpolate_points(
                self.points[i],
                self.points[i + 1],
                self.step_m,
            )

            if i > 0:
                segment = segment[1:]

            full_route.extend(segment)

        return full_route

    # =====================================================
    # SAVE ROUTE
    # =====================================================

    @staticmethod
    def save_route(route, filename):
        os.makedirs(
            os.path.dirname(filename),
            exist_ok=True
        )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:
            for lat, lon in route:
                f.write(f"{lat},{lon}\n")
