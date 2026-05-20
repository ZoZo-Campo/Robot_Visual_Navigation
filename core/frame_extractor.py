import os
import cv2


class FrameExtractor:
    def __init__(
        self,
        video_path,
        output_dir,
        fps=1,
        frame_step_seconds=2,
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.fps = fps
        self.frame_step_seconds = frame_step_seconds

    def extract_frames(
        self,
        progress_callback=None,
        clear_output=True,
    ):
        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        if clear_output:
            for file in os.listdir(
                self.output_dir
            ):
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    os.remove(
                        os.path.join(
                            self.output_dir,
                            file
                        )
                    )

        cap = cv2.VideoCapture(
            self.video_path
        )

        if not cap.isOpened():
            raise ValueError(
                f"Cannot open video: {self.video_path}"
            )

        video_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if video_fps <= 0:
            raise ValueError(
                "Invalid video FPS."
            )

        frame_interval = max(
            1,
            int(video_fps * self.frame_step_seconds)
        )

        saved_count = 0
        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % frame_interval == 0:
                filename = os.path.join(
                    self.output_dir,
                    f"frame_{saved_count:05d}.jpg"
                )
                h = frame.shape[0]

                frame = frame[
                    0:int(h * 0.75),
                    :
                ]

                cv2.imwrite(
                    filename,
                    frame
                )

                saved_count += 1

            frame_index += 1

            if progress_callback is not None:
                progress_callback(
                    current=frame_index,
                    total=total_frames,
                    saved=saved_count,
                )

        cap.release()

        return {
            "video_fps": video_fps,
            "total_frames": total_frames,
            "saved_frames": saved_count,
            "frame_step_seconds": self.frame_step_seconds,
            "output_dir": self.output_dir,
        }