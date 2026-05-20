import os
import sys
import csv
import time

import streamlit as st
import folium
from streamlit_folium import st_folium

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)

from config import *
from matching.dino_matcher import DINOMatcher
from core.route_planner import RoutePlanner
from core.streetview_client import StreetViewClient
from core.frame_extractor import FrameExtractor
from core.gps_utils import same_point
from core.replay_localizer import ReplayLocalizer
from matching.orb_matcher import ORBMatcher
from matching.cnn_matcher import CNNMatcher
from matching.hybrid_matcher import HybridMatcher

from replay.replay_matcher import ReplayMatcher


# =========================================================
# PATHS
# =========================================================

DATA_DIR = os.path.join(BASE_DIR, "data")

ROUTES_DIR = os.path.join(
    DATA_DIR,
    "routes"
)

STREETVIEW_DIR = os.path.join(
    DATA_DIR,
    "streetview"
)

STREETVIEW_IMAGES_DIR = os.path.join(
    STREETVIEW_DIR,
    "images"
)

METADATA_FILE = os.path.join(
    STREETVIEW_DIR,
    "metadata.csv"
)

GPS_ROUTE_FILE = os.path.join(
    ROUTES_DIR,
    "gps_route.txt"
)

ROBOT_PHOTOS_DIR = os.path.join(
    DATA_DIR,
    "robot_inputs",
    "photos"
)

ROBOT_VIDEOS_DIR = os.path.join(
    DATA_DIR,
    "robot_inputs",
    "videos"
)

FRAMES_DIR = os.path.join(
    DATA_DIR,
    "frames"
)

RESULTS_DIR = os.path.join(
    DATA_DIR,
    "results"
)

REPLAY_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "replay_results.csv"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

LOGO_PATH = os.path.join(
    ASSETS_DIR,
    "logo-cvut.jpg"
)

for directory in [
    ROUTES_DIR,
    STREETVIEW_IMAGES_DIR,
    ROBOT_PHOTOS_DIR,
    ROBOT_VIDEOS_DIR,
    FRAMES_DIR,
    RESULTS_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True
    )


DEFAULT_MAP_CENTER = globals().get(
    "DEFAULT_CENTER",
    [50.104493, 14.394761]
)

DEFAULT_ZOOM_LEVEL = globals().get(
    "DEFAULT_ZOOM",
    15
)


# =========================================================
# HELPERS
# =========================================================

def save_route_file(route, filename):
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


def read_route_file(filename):
    route = []

    if not os.path.exists(filename):
        return route

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            lat, lon = line.split(",")

            route.append(
                (
                    float(lat),
                    float(lon)
                )
            )

    return route


def load_metadata(metadata_file):
    metadata = {}

    if not os.path.exists(metadata_file):
        return metadata

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_name = os.path.basename(
                row.get("image_file", "")
            )

            metadata[image_name] = row

    return metadata


def get_image_files(directory):
    if not os.path.exists(directory):
        return []

    return sorted(
        file for file in os.listdir(directory)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    )


def get_video_files(directory):
    if not os.path.exists(directory):
        return []

    return sorted(
        file for file in os.listdir(directory)
        if file.lower().endswith(
            (".mp4", ".mov", ".m4v")
        )
    )


def get_map_center():
    if (
        st.session_state.start_point is not None
        and st.session_state.end_point is not None
    ):
        return [
            (
                st.session_state.start_point[0]
                + st.session_state.end_point[0]
            ) / 2,
            (
                st.session_state.start_point[1]
                + st.session_state.end_point[1]
            ) / 2,
        ]

    if st.session_state.start_point is not None:
        return st.session_state.start_point

    if st.session_state.end_point is not None:
        return st.session_state.end_point

    return DEFAULT_MAP_CENTER
def draw_navigation_map():
    m = folium.Map(
        location=get_map_center(),
        zoom_start=DEFAULT_ZOOM_LEVEL
    )

    if st.session_state.route is not None:
        folium.PolyLine(
            st.session_state.route,
            color="blue",
            weight=5,
            opacity=0.9,
            popup="Planned route",
        ).add_to(m)

    if st.session_state.replay_positions:
        folium.PolyLine(
            st.session_state.replay_positions,
            color="purple",
            weight=4,
            opacity=0.8,
            popup="Estimated replay trajectory",
        ).add_to(m)

    if st.session_state.start_point is not None:
        folium.Marker(
            st.session_state.start_point,
            popup="Start",
            icon=folium.Icon(color="green"),
        ).add_to(m)

    for i, waypoint in enumerate(
        st.session_state.waypoints
    ):
        folium.Marker(
            waypoint,
            popup=f"Waypoint {i + 1}",
            icon=folium.Icon(color="orange"),
        ).add_to(m)

    if st.session_state.end_point is not None:
        folium.Marker(
            st.session_state.end_point,
            popup="End",
            icon=folium.Icon(color="red"),
        ).add_to(m)

    if st.session_state.estimated_position is not None:
        folium.Marker(
            st.session_state.estimated_position,
            popup="Estimated position",
            icon=folium.Icon(color="purple"),
        ).add_to(m)

    map_data = st_folium(
        m,
        width=1200,
        height=650,
        returned_objects=["last_clicked"]
    )

    return map_data


def create_matchers():
    orb = ORBMatcher(
        n_features=ORB_FEATURES,
        distance_threshold=ORB_DISTANCE_THRESHOLD,
    )

    cnn = CNNMatcher(
        image_size=CNN_IMAGE_SIZE
    )

    hybrid = HybridMatcher(
        orb_weight=ORB_WEIGHT,
        cnn_weight=CNN_WEIGHT,
    )

    return orb, cnn, hybrid


def run_matching(
    image_path,
    mode,
    database_dir,
    top_k=5,
    progress_callback=None,
):
    if mode == "ORB":
        orb = ORBMatcher(
            n_features=ORB_FEATURES,
            distance_threshold=ORB_DISTANCE_THRESHOLD,
        )

        return orb.search(
            image_path,
            database_dir,
            top_k=top_k,
            progress_callback=progress_callback,
        )

    if mode == "CNN":
        cnn = CNNMatcher(
            image_size=CNN_IMAGE_SIZE
        )

        return cnn.search(
            image_path,
            database_dir,
            top_k=top_k,
            progress_callback=progress_callback,
        )
    if mode == "DINO":
        dino = DINOMatcher()

        return dino.search(
            image_path,
            database_dir,
            top_k=top_k,
            progress_callback=progress_callback,
        )
    orb, cnn, hybrid = create_matchers()

    orb_results = orb.search(
        image_path,
        database_dir,
        top_k=999999,
    )

    cnn_results = cnn.search(
        image_path,
        database_dir,
        top_k=999999,
    )

    return hybrid.fuse_results(
        orb_results,
        cnn_results,
        top_k=top_k,
    )


def attach_gps_to_result(result, metadata):
    filename = result.get("filename")

    meta = metadata.get(filename)

    if meta is None:
        result["lat"] = None
        result["lon"] = None
        return result

    result["lat"] = float(
        meta["target_lat"]
    )

    result["lon"] = float(
        meta["target_lon"]
    )

    return result


def display_top_results(
    results,
    metadata,
):
    if not results:
        st.warning("No match found.")
        return

    best = attach_gps_to_result(
        results[0],
        metadata
    )

    st.success(
        f"Best match: {best['filename']} — "
        f"{best['score']:.2f}%"
    )

    if best["lat"] is not None:
        st.session_state.estimated_position = [
            best["lat"],
            best["lon"]
        ]

        st.info(
            f"Estimated GPS position: "
            f"{best['lat']}, {best['lon']}"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Query image")

        if st.session_state.current_query_image is not None:
            st.image(
                st.session_state.current_query_image,
                width=450
            )

    with col2:
        st.subheader("Best Street View match")

        best_path = os.path.join(
            STREETVIEW_IMAGES_DIR,
            best["filename"]
        )

        if os.path.exists(best_path):
            st.image(
                best_path,
                width=450,
                caption=best["filename"]
            )

    st.subheader("Top results")

    for result in results:
        line = (
            f"{result['filename']} → "
            f"{result['score']:.2f}%"
        )

        if "orb_score" in result:
            line += (
                f" | ORB {result['orb_score']:.2f}%"
                f" | CNN {result['cnn_score']:.2f}%"
            )

        if "good_matches" in result:
            line += (
                f" | matches {result['good_matches']}"
            )

        st.write(line)


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Robot Visual Navigation V2",
    layout="wide"
)

if os.path.exists(LOGO_PATH):
    st.sidebar.image(
        LOGO_PATH,
        width=160
    )

st.title("Robot Visual Navigation V2")

st.write(
    "Route planning, Street View acquisition, "
    "visual matching and offline replay localization."
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "start_point": None,
    "end_point": None,
    "waypoints": [],
    "route": None,
    "last_clicked_point": None,
    "estimated_position": None,
    "replay_positions": [],
    "current_query_image": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Global settings")

selection_mode = st.sidebar.radio(
    "Point input mode",
    [
        "Map selection",
        "GPS coordinates"
    ]
)

step_m = st.sidebar.number_input(
    "Route sampling step (m)",
    min_value=1,
    max_value=20,
    value=int(STEP_M)
)

network_type = st.sidebar.selectbox(
    "OSM network type",
    [
        "walk",
        "drive",
        "bike",
        "all"
    ],
    index=0
)

heading_mode = st.sidebar.selectbox(
    "Street View heading mode",
    [
        "route",
        "north"
    ],
    index=0
)

matching_mode_global = st.sidebar.selectbox(
    "Default matching mode",
    [
        "ORB",
        "CNN",
        "DINO",
        "Hybrid"
    ],
    index=2
)


# =========================================================
# TABS
# =========================================================

(
    tab_route,
    tab_streetview,
    tab_single_match,
    tab_replay,
    tab_exports,
) = st.tabs(
    [
        "1. Route",
        "2. Street View",
        "3. Single Image",
        "4. Replay",
        "5. Exports",
    ]
)

# =========================================================
# TAB 1 — ROUTE
# =========================================================

with tab_route:

    st.header("1. Route builder")

    c1, c2 = st.columns(2)

    with c1:
        st.write("Start point")
        st.code(st.session_state.start_point)

    with c2:
        st.write("End point")
        st.code(st.session_state.end_point)

    if st.session_state.waypoints:
        st.write("Waypoints")

        for i, point in enumerate(
            st.session_state.waypoints
        ):
            st.code(
                f"{i + 1}: {point}"
            )

    # =====================================================
    # GPS INPUT
    # =====================================================

    if selection_mode == "GPS coordinates":

        st.subheader(
            "Manual GPS input"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            start_lat = st.text_input(
                "Start latitude"
            )

        with c2:
            start_lon = st.text_input(
                "Start longitude"
            )

        with c3:
            end_lat = st.text_input(
                "End latitude"
            )

        with c4:
            end_lon = st.text_input(
                "End longitude"
            )

        if st.button(
            "Validate GPS coordinates"
        ):
            try:

                start = [
                    float(start_lat),
                    float(start_lon)
                ]

                end = [
                    float(end_lat),
                    float(end_lon)
                ]

                if same_point(
                    start,
                    end
                ):
                    st.error(
                        "Start and end points "
                        "cannot be identical."
                    )

                else:
                    st.session_state.start_point = (
                        start
                    )

                    st.session_state.end_point = (
                        end
                    )

                    st.session_state.route = None

                    st.rerun()

            except ValueError:

                st.error(
                    "Please enter valid coordinates."
                )

    # =====================================================
    # MAP
    # =====================================================

    map_data = draw_navigation_map()

    # =====================================================
    # MAP SELECTION
    # =====================================================

    if selection_mode == "Map selection":

        clicked = None

        if map_data is not None:
            clicked = map_data.get(
                "last_clicked"
            )

        if clicked is not None:

            st.session_state.last_clicked_point = [
                clicked["lat"],
                clicked["lng"]
            ]

        if (
            st.session_state.last_clicked_point
            is not None
        ):

            st.write(
                f"Selected point: "
                f"{st.session_state.last_clicked_point}"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                if st.button(
                    "Set as start point",
                    use_container_width=True
                ):

                    st.session_state.start_point = (
                        st.session_state.last_clicked_point
                    )

                    st.session_state.route = None

                    st.rerun()

            with c2:

                if st.button(
                    "Add waypoint",
                    use_container_width=True
                ):

                    st.session_state.waypoints.append(
                        st.session_state.last_clicked_point
                    )

                    st.session_state.route = None

                    st.rerun()

            with c3:

                if st.button(
                    "Set as end point",
                    use_container_width=True
                ):

                    point = (
                        st.session_state.last_clicked_point
                    )

                    if same_point(
                        point,
                        st.session_state.start_point
                    ):

                        st.error(
                            "Start and end points "
                            "cannot be identical."
                        )

                    else:

                        st.session_state.end_point = (
                            point
                        )

                        st.session_state.route = None

                        st.rerun()

    st.divider()

    c1, c2, c3 = st.columns(3)

    # =====================================================
    # CREATE ROUTE
    # =====================================================

    with c1:

        if st.button(
            "Create route",
            use_container_width=True
        ):

            if (
                st.session_state.start_point is None
                or
                st.session_state.end_point is None
            ):

                st.error(
                    "Please define "
                    "start and end points."
                )

            else:

                ordered_points = (
                    [st.session_state.start_point]
                    +
                    st.session_state.waypoints
                    +
                    [st.session_state.end_point]
                )

                with st.spinner(
                    "Computing route..."
                ):

                    planner = RoutePlanner(
                        points=ordered_points,
                        network_type=network_type,
                        step_m=step_m,
                    )

                    route = (
                        planner.compute_route()
                    )

                    save_route_file(
                        route,
                        GPS_ROUTE_FILE
                    )

                    st.session_state.route = (
                        route
                    )

                st.rerun()

    # =====================================================
    # REMOVE WAYPOINT
    # =====================================================

    with c2:

        if st.button(
            "Remove last waypoint",
            use_container_width=True
        ):

            if st.session_state.waypoints:

                st.session_state.waypoints.pop()

                st.session_state.route = None

                st.rerun()

    # =====================================================
    # CLEAR
    # =====================================================

    with c3:

        if st.button(
            "Clear route",
            use_container_width=True
        ):

            st.session_state.start_point = None

            st.session_state.end_point = None

            st.session_state.waypoints = []

            st.session_state.route = None

            st.session_state.replay_positions = []

            st.session_state.last_clicked_point = None

            st.rerun()

    if st.session_state.route is not None:

        st.success(
            f"Route created with "
            f"{len(st.session_state.route)} GPS points."
        )


# =========================================================
# TAB 2 — STREET VIEW
# =========================================================

with tab_streetview:

    st.header(
        "2. Street View database"
    )

    if st.session_state.route is None:

        st.warning(
            "Create a route first."
        )

    else:

        st.write(
            f"Route points: "
            f"{len(st.session_state.route)}"
        )

        if st.button(
            "Download Street View database",
            use_container_width=True
        ):

            progress_bar = st.progress(0)

            status_text = st.empty()

            stats_text = st.empty()

            start_time = time.time()

            def update_progress(
                current,
                total,
                downloaded,
                skipped,
                message,
            ):

                progress = (
                    current / total
                    if total else 0
                )

                progress_bar.progress(
                    progress
                )

                elapsed = (
                    time.time()
                    - start_time
                )

                remaining = 0

                if current > 0:

                    estimated_total = (
                        elapsed / current
                    ) * total

                    remaining = (
                        estimated_total
                        - elapsed
                    )

                status_text.write(
                    message
                )

                stats_text.write(
                    f"Progress: {current}/{total} | "
                    f"Downloaded: {downloaded} | "
                    f"Skipped: {skipped} | "
                    f"Remaining: {remaining:.1f}s"
                )

            with st.spinner(
                "Downloading Street View images..."
            ):

                client = StreetViewClient(
                    api_key=API_KEY,
                    image_size=IMAGE_SIZE,
                    fov=FOV,
                    pitch=PITCH,
                    radius=RADIUS,
                    max_distance_to_pano=MAX_DISTANCE_TO_PANO,
                    heading_mode=heading_mode,
                )

                result = (
                    client.acquire_from_route(
                        route=st.session_state.route,
                        output_dir=STREETVIEW_IMAGES_DIR,
                        metadata_file=METADATA_FILE,
                        progress_callback=update_progress,
                    )
                )

            st.success(
                f"Street View acquisition completed: "
                f"{result['downloaded']} downloaded, "
                f"{result['skipped']} skipped."
            )

    image_files = get_image_files(
        STREETVIEW_IMAGES_DIR
    )

    if image_files:

        st.subheader(
            "Street View preview"
        )

        cols = st.columns(4)

        for i, filename in enumerate(
            image_files[:8]
        ):

            with cols[i % 4]:

                st.image(
                    os.path.join(
                        STREETVIEW_IMAGES_DIR,
                        filename
                    ),
                    caption=filename,
                    use_container_width=True
                )


# =========================================================
# TAB 3 — SINGLE IMAGE
# =========================================================

with tab_single_match:

    st.header(
        "3. Single image localization"
    )

    uploaded_image = st.file_uploader(
        "Upload robot image",
        type=["jpg", "jpeg", "png"]
    )

    query_image_path = os.path.join(
        ROBOT_PHOTOS_DIR,
        "query_image.jpg"
    )

    if uploaded_image is not None:

        with open(
            query_image_path,
            "wb"
        ) as f:

            f.write(
                uploaded_image.getbuffer()
            )

        st.session_state.current_query_image = (
            query_image_path
        )

        st.success(
            "Query image saved."
        )

    if (
        st.session_state.current_query_image
        is not None
    ):

        st.image(
            st.session_state.current_query_image,
            width=500
        )

    matching_mode = st.selectbox(
        "Matching method",
        [
            "ORB",
            "CNN",
            "Hybrid"
        ],
        index=2
    )

    if st.button(
        "Run localization",
        use_container_width=True
    ):

        if (
            st.session_state.current_query_image
            is None
        ):

            st.error(
                "Upload an image first."
            )

        else:

            metadata = load_metadata(
                METADATA_FILE
            )

            with st.spinner(
                "Running matching..."
            ):

                results = run_matching(
                    image_path=st.session_state.current_query_image,
                    mode=matching_mode,
                    database_dir=STREETVIEW_IMAGES_DIR,
                    top_k=5,
                )

            display_top_results(
                results,
                metadata
            )


# =========================================================
# TAB 4 — REPLAY
# =========================================================

with tab_replay:

    st.header(
        "4. Offline replay localization"
    )

    uploaded_video = st.file_uploader(
        "Upload robot video 1fps toutes les 2 secondes",
        type=["mp4", "mov", "m4v"]
    )

    if uploaded_video is not None:

        video_path = os.path.join(
            ROBOT_VIDEOS_DIR,
            uploaded_video.name
        )

        with open(
            video_path,
            "wb"
        ) as f:

            f.write(
                uploaded_video.getbuffer()
            )

        st.success(
            "Video uploaded."
        )

        extraction_fps = st.slider(
            "Frame extraction FPS",
            min_value=1,
            max_value=10,
            value=1
        )

        if st.button(
            "Extract frames",
            use_container_width=True
        ):

            progress_bar = st.progress(0)

            status = st.empty()

            extractor = FrameExtractor(
                video_path=video_path,
                output_dir=FRAMES_DIR,
                fps=extraction_fps,
            )

            def frame_progress(
                current,
                total,
                saved,
            ):

                progress_bar.progress(
                    current / total
                )

                status.write(
                    f"Frames processed: "
                    f"{current}/{total} | "
                    f"Saved: {saved}"
                )

            result = extractor.extract_frames(
                progress_callback=frame_progress
            )

            st.success(
                f"{result['saved_frames']} frames extracted."
            )

    frame_files = get_image_files(
        FRAMES_DIR
    )

    if frame_files:

        st.write(
            f"Frames available: "
            f"{len(frame_files)}"
        )

        replay_matching_mode = st.selectbox(
            "Replay matching mode",
            [
                "ORB",
                "CNN",
                "DINO",
                "Hybrid"
            ],
            index=2
        )

        if st.button(
            "Run replay localization",
            use_container_width=True
        ):

            progress_bar = st.progress(0)

            status = st.empty()

            st.session_state.replay_positions = []

            localizer = ReplayLocalizer(
                database_dir=STREETVIEW_IMAGES_DIR,
                metadata_file=METADATA_FILE,
                matching_mode=replay_matching_mode,
                top_k=5,
                orb_features=ORB_FEATURES,
                orb_distance_threshold=ORB_DISTANCE_THRESHOLD,
                cnn_image_size=CNN_IMAGE_SIZE,
                orb_weight=ORB_WEIGHT,
                cnn_weight=CNN_WEIGHT,
                search_window=3,
                max_jump=3,
                min_score=40,
            )

            def replay_progress(
                current,
                total,
                remaining,
                filename,
            ):

                progress_bar.progress(
                    current / total
                )

                status.write(
                    f"Replay progress: "
                    f"{current}/{total} | "
                    f"ETA: {remaining:.1f}s | "
                    f"{filename}"
                )

            replay_results = localizer.localize_frames(
                frames_dir=FRAMES_DIR,
                progress_callback=replay_progress,
            )

            for result in replay_results:

                if (
                    result["lat"] is not None
                    and
                    result["lon"] is not None
                ):

                    st.session_state.replay_positions.append(
                        [
                            result["lat"],
                            result["lon"]
                        ]
                    )

            ReplayMatcher.save_results(
                replay_results,
                REPLAY_RESULTS_FILE,
            )

            st.success(
                "Replay localization completed."
            )

            st.write(
                f"Valid localized frames: "
                f"{len(replay_results)}"
            )
# =========================================================
# TAB 5 — EXPORTS
# =========================================================

with tab_exports:

    st.header("5. Exports")

    if os.path.exists(
        GPS_ROUTE_FILE
    ):

        with open(
            GPS_ROUTE_FILE,
            "rb"
        ) as f:

            st.download_button(
                "Download GPS route",
                data=f,
                file_name="gps_route.txt"
            )

    if os.path.exists(
        METADATA_FILE
    ):

        with open(
            METADATA_FILE,
            "rb"
        ) as f:

            st.download_button(
                "Download metadata",
                data=f,
                file_name="metadata.csv"
            )

    if os.path.exists(
        REPLAY_RESULTS_FILE
    ):

        with open(
            REPLAY_RESULTS_FILE,
            "rb"
        ) as f:

            st.download_button(
                "Download replay results",
                data=f,
                file_name="replay_results.csv"
            )

    st.write("Project root:")

    st.code(BASE_DIR)