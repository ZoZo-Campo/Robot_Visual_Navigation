import os

import osmnx as ox
import networkx as nx

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
    ):
        self.points = points
        self.network_type = network_type
        self.step_m = step_m
        self.padding_m = padding_m

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

        return center_point, download_radius

    # =====================================================
    # BUILD GRAPH
    # =====================================================

    def build_graph(self):
        center_point, download_radius = (
            self.get_graph_center_and_radius()
        )

        print(f"OSM center: {center_point}")
        print(f"OSM radius: {download_radius} m")

        graph = ox.graph_from_point(
            center_point,
            dist=download_radius,
            network_type=self.network_type,
            simplify=True,
        )

        return graph

    # =====================================================
    # CLOSEST NODE
    # =====================================================

    @staticmethod
    def nearest_node(graph, point):
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

        graph = self.build_graph()

        full_route = []

        for i in range(len(self.points) - 1):
            segment = self.compute_segment(
                graph=graph,
                start_point=self.points[i],
                end_point=self.points[i + 1],
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