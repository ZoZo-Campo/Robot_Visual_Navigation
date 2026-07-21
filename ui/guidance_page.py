"""Streamlit page for V4 offline visual-guidance simulation."""

import csv
import os

import cv2
import streamlit as st

from guidance.path_follower import GuidanceConfig, VisualPathFollower
from guidance.streetview_guidance import StreetViewGuidanceBuilder
from guidance.video_processor import GuidanceVideoProcessor


def _video_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        filename
        for filename in os.listdir(directory)
        if filename.lower().endswith((".mp4", ".mov", ".m4v"))
    )


def _save_upload(uploaded_video, directory):
    os.makedirs(directory, exist_ok=True)
    safe_name = os.path.basename(uploaded_video.name)
    destination = os.path.join(directory, safe_name)
    with open(destination, "wb") as video_file:
        video_file.write(uploaded_video.getbuffer())
    return destination


def _load_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _guidance_config(defaults):
    return GuidanceConfig(
        lateral_gain=defaults["lateral_gain"],
        heading_gain=defaults["heading_gain"],
        max_angular_z=defaults["max_angular_z"],
        deadband=defaults["deadband"],
        nominal_linear_x=defaults["nominal_linear_x"],
        min_confidence=defaults["min_confidence"],
        reference_alignment_gain=defaults["reference_alignment_gain"],
        route_turn_gain=defaults["route_turn_gain"],
        local_stability_gain=defaults["local_stability_gain"],
        require_streetview_reference=True,
    )


def _build_video_targets(
    video_path,
    frames_dir,
    database_dir,
    metadata_file,
    matching_mode,
    config,
    defaults,
    status_callback=None,
):
    from core.frame_extractor import FrameExtractor
    from core.replay_localizer import ReplayLocalizer

    if status_callback is not None:
        status_callback("Extracting VPR anchor frames from the video...")
    extractor = FrameExtractor(
        video_path=video_path,
        output_dir=frames_dir,
        fps=1,
        frame_step_seconds=defaults["reference_interval_s"],
    )
    extractor.extract_frames()

    if status_callback is not None:
        status_callback("Matching anchor frames with the Street View database...")
    localizer = ReplayLocalizer(
        database_dir=database_dir,
        metadata_file=metadata_file,
        matching_mode=matching_mode,
        top_k=5,
        min_score=0,
    )
    localization_results = localizer.localize_frames(frames_dir)

    if status_callback is not None:
        status_callback("Building GPS, heading and Street View guidance targets...")
    builder = StreetViewGuidanceBuilder(
        database_dir=database_dir,
        metadata_file=metadata_file,
        guidance_config=config,
        route_lookahead_images=defaults["route_lookahead_images"],
    )
    return (
        builder.build_targets(localization_results),
        localization_results,
        localizer.database_summary,
    )


def render_guidance_tab(
    video_dir,
    results_dir,
    frames_dir,
    database_dir,
    metadata_file,
    matching_mode,
    database_label,
    defaults,
):
    st.header("5. Street View DB-guided navigation")
    st.write(
        "The robot video is first localized against the active Street View database. "
        "The matched GPS position and route-facing reference image then guide the "
        "visual path controller."
    )
    st.info(
        "ROS convention: angular_z > 0 turns left, angular_z < 0 turns right. "
        "No command is sent to the real robot in this offline simulation."
    )
    st.success(f"Guidance database: {database_label}")

    local_videos = _video_files(video_dir)
    source_mode = st.radio(
        "Video source",
        ["Local test video", "Upload a video"],
        horizontal=True,
        key="guidance_video_source",
    )
    selected_video_path = None

    if source_mode == "Local test video":
        if local_videos:
            selected_name = st.selectbox(
                "Available video",
                local_videos,
                key="guidance_local_video",
            )
            selected_video_path = os.path.join(video_dir, selected_name)
        else:
            st.warning("No local test video is available.")
    else:
        uploaded = st.file_uploader(
            "Upload the robot camera video",
            type=["mp4", "mov", "m4v"],
            key="guidance_uploaded_video",
        )
        if uploaded is not None:
            selected_video_path = _save_upload(uploaded, video_dir)
            st.success(f"Video saved: {os.path.basename(selected_video_path)}")

    with st.expander("Controller settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            defaults["lateral_gain"] = st.slider(
                "Lateral gain", 0.0, 2.0, float(defaults["lateral_gain"]), 0.05
            )
            defaults["heading_gain"] = st.slider(
                "Heading gain", 0.0, 2.0, float(defaults["heading_gain"]), 0.05
            )
        with c2:
            defaults["max_angular_z"] = st.slider(
                "Maximum angular speed (rad/s)",
                0.1,
                1.5,
                float(defaults["max_angular_z"]),
                0.05,
            )
            defaults["nominal_linear_x"] = st.slider(
                "Nominal linear speed (m/s)",
                0.0,
                1.0,
                float(defaults["nominal_linear_x"]),
                0.05,
            )
        with c3:
            defaults["deadband"] = st.slider(
                "Straight deadband", 0.0, 0.2, float(defaults["deadband"]), 0.005
            )
            defaults["min_confidence"] = st.slider(
                "Minimum confidence", 0.0, 1.0, float(defaults["min_confidence"]), 0.05
            )

        st.caption("Street View / local-path fusion")
        r1, r2, r3 = st.columns(3)
        with r1:
            defaults["reference_alignment_gain"] = st.slider(
                "Street View alignment gain",
                0.0,
                2.0,
                float(defaults["reference_alignment_gain"]),
                0.05,
            )
        with r2:
            defaults["route_turn_gain"] = st.slider(
                "GPS route turn gain",
                0.0,
                2.0,
                float(defaults["route_turn_gain"]),
                0.05,
            )
        with r3:
            defaults["local_stability_gain"] = st.slider(
                "Local path stability gain",
                0.0,
                1.0,
                float(defaults["local_stability_gain"]),
                0.05,
            )

    processing_fps = st.slider(
        "Analysis frequency (frames/s)",
        min_value=2,
        max_value=30,
        value=10,
        help="10 FPS is sufficient for the offline test and keeps processing fast.",
    )

    if selected_video_path is not None:
        st.video(selected_video_path)
        preview_col, run_col = st.columns(2)

        with preview_col:
            preview_clicked = st.button(
                "Analyse first frame with Street View DB",
                width="stretch",
                key="guidance_preview_button",
            )
        with run_col:
            run_clicked = st.button(
                "Run complete guidance simulation",
                width="stretch",
                type="primary",
                key="guidance_run_button",
            )

        config = _guidance_config(defaults)
        if preview_clicked:
            capture = cv2.VideoCapture(selected_video_path)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                st.error("The first video frame could not be read.")
            else:
                os.makedirs(results_dir, exist_ok=True)
                preview_path = os.path.join(results_dir, "guidance_preview.jpg")
                cv2.imwrite(preview_path, frame)
                try:
                    from core.replay_localizer import ReplayLocalizer

                    preview_localizer = ReplayLocalizer(
                        database_dir=database_dir,
                        metadata_file=metadata_file,
                        matching_mode=matching_mode,
                        top_k=5,
                        min_score=0,
                    )
                    preview_match = preview_localizer.localize_frame(preview_path)
                    preview_builder = StreetViewGuidanceBuilder(
                        database_dir=database_dir,
                        metadata_file=metadata_file,
                        guidance_config=config,
                        route_lookahead_images=defaults["route_lookahead_images"],
                    )
                    preview_target = preview_builder.build_target(preview_match or {})
                    result = VisualPathFollower(config).process(
                        frame,
                        reference_target=preview_target,
                    )
                except Exception as exc:
                    st.error("Street View DB preview failed.")
                    st.exception(exc)
                    result = None

                if result is not None:
                    robot_col, reference_col = st.columns(2)
                    with robot_col:
                        st.image(
                            cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                            caption=(
                                f"{result.instruction} | angular_z="
                                f"{result.angular_z:+.3f} rad/s"
                            ),
                            width="stretch",
                        )
                    with reference_col:
                        reference_path = os.path.join(
                            database_dir,
                            result.reference_filename,
                        )
                        if os.path.exists(reference_path):
                            st.image(
                                reference_path,
                                caption=(
                                    f"Matched DB image: {result.reference_filename} | "
                                    f"GPS {result.reference_latitude}, "
                                    f"{result.reference_longitude}"
                                ),
                                width="stretch",
                            )

        if run_clicked:
            os.makedirs(results_dir, exist_ok=True)
            output_video = os.path.join(results_dir, "guidance_db_annotated.mp4")
            output_csv = os.path.join(results_dir, "guidance_db_commands.csv")
            progress = st.progress(0)
            status = st.empty()

            def update_progress(current, total, processed):
                progress.progress(current / total if total else 0.0)
                status.write(
                    f"Input frames: {current}/{total} | Commands generated: {processed}"
                )

            try:
                with st.spinner(
                    "Localizing with MixVPR, comparing Street View references and "
                    "generating commands..."
                ):
                    reference_targets, localization_results, database_summary = (
                        _build_video_targets(
                            selected_video_path,
                            frames_dir,
                            database_dir,
                            metadata_file,
                            matching_mode,
                            config,
                            defaults,
                            status_callback=status.write,
                        )
                    )
                    summary = GuidanceVideoProcessor(config).process_video(
                        selected_video_path,
                        output_video,
                        output_csv,
                        processing_fps=processing_fps,
                        reference_targets=reference_targets,
                        reference_interval_s=defaults["reference_interval_s"],
                        progress_callback=update_progress,
                    )
                    summary["localization_results"] = localization_results
                    summary["database_summary"] = database_summary
                st.session_state.guidance_summary = summary
            except Exception as exc:
                st.error("Guidance simulation failed.")
                st.exception(exc)
            else:
                st.success(
                    f"Simulation completed: {summary['processed_frames']} commands generated."
                )

    summary = st.session_state.get("guidance_summary")
    if summary and os.path.exists(summary.get("output_video", "")):
        st.subheader("Guidance result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Commands", summary["processed_frames"])
        c2.metric("Mean confidence", f"{summary['mean_confidence']:.2f}")
        c3.metric("Mean |angular_z|", f"{summary['mean_abs_angular_z']:.3f} rad/s")
        c4.metric("DB-guided frames", f"{summary['db_guided_ratio'] * 100:.1f}%")
        st.video(summary["output_video"])

        rows = _load_rows(summary["output_csv"])
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)

        with open(summary["output_video"], "rb") as video_file:
            st.download_button(
                "Download annotated guidance video",
                data=video_file,
                file_name="guidance_db_annotated.mp4",
                mime="video/mp4",
            )
        with open(summary["output_csv"], "rb") as csv_file:
            st.download_button(
                "Download numerical velocity commands (CSV)",
                data=csv_file,
                file_name="guidance_db_commands.csv",
                mime="text/csv",
            )
