import cv2
import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class AnimationCreator:
    def __init__(
        self,
        video_path,
        start_frame,
        end_frame,
        image_shape,
        padding=50,
        aspect_ratio=(9, 16),
    ):
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.image_shape = image_shape
        self.padding = padding
        self.aspect_ratio = aspect_ratio
        self.frames = []
        self.tracking_positions = []
        print("AnimationCreator created")

    @staticmethod
    def make_divisible_by_8(value):
        """Корректируем значение, чтобы оно стало кратным 8."""
        return (value + 7) // 8 * 8

    def load_video_frames(self):
        """Loads frames from the video within the specified range."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > self.end_frame:
                break
            self.frames.append(frame)

        cap.release()
        self.frames = np.array(self.frames)
        print(f"{self.frames.shape=}")

    def calculate_fixed_crop_area(self, landmarks_data):
        """Calculates the fixed crop area based on landmarks and image shape."""
        width, height = self.image_shape
        scaled_landmarks = landmarks_data.numpy().copy()
        scaled_landmarks[:, :, 0] *= width
        scaled_landmarks[:, :, 1] *= height

        valid_x = scaled_landmarks[:, :, 0][scaled_landmarks[:, :, 0] != 0]
        valid_y = scaled_landmarks[:, :, 1][scaled_landmarks[:, :, 1] != 0]

        min_x, max_x = np.min(valid_x) - self.padding, np.max(valid_x) + self.padding
        min_y, max_y = np.min(valid_y) - self.padding, np.max(valid_y) + self.padding

        crop_width = max_x - min_x
        crop_height = max(
            crop_width * self.aspect_ratio[1] // self.aspect_ratio[0], max_y - min_y
        )

        crop_height = self.make_divisible_by_8(crop_height)
        crop_width = self.make_divisible_by_8(crop_width)

        print(f"{crop_width=}")
        print(f"{crop_height=}")
        return crop_width, crop_height

    def get_tracking_positions(self, landmarks_data, crop_width, crop_height):
        """Calculates the tracking positions for cropping based on the landmarks data."""
        width, height = self.image_shape
        scaled_landmarks = landmarks_data.numpy().copy()
        scaled_landmarks[:, :, 0] *= width
        scaled_landmarks[:, :, 1] *= height

        for frame_landmarks in scaled_landmarks:
            valid_x = frame_landmarks[:, 0][frame_landmarks[:, 0] != 0]
            valid_y = frame_landmarks[:, 1][frame_landmarks[:, 1] != 0]

            if len(valid_x) > 0 and len(valid_y) > 0:
                center_x = int((np.min(valid_x) + np.max(valid_x)) // 2)
                center_y = int((np.min(valid_y) + np.max(valid_y)) // 2)

                crop_min_x = max(0, center_x - crop_width // 2)
                crop_max_x = min(width, center_x + crop_width // 2)
                crop_min_y = max(0, center_y - crop_height // 2)
                crop_max_y = min(height, center_y + crop_height // 2)

                self.tracking_positions.append(
                    (int(crop_min_x), int(crop_min_y), int(crop_max_x), int(crop_max_y))
                )
            else:
                self.tracking_positions.append(
                    self.tracking_positions[-1]
                    if self.tracking_positions
                    else (0, 0, crop_width, crop_height)
                )

        print(f"{len(self.tracking_positions)=}")

    def smooth_crop_positions(self, expansion_factor=3):
        """Smoothens crop positions by interpolating them over frames."""
        min_x_vals = [crop[0] for crop in self.tracking_positions]
        min_y_vals = [crop[1] for crop in self.tracking_positions]
        max_x_vals = [crop[2] for crop in self.tracking_positions]
        max_y_vals = [crop[3] for crop in self.tracking_positions]

        original_indices = np.arange(len(self.tracking_positions))
        interpolated_indices = np.linspace(
            0,
            len(self.tracking_positions) - 1,
            num=len(self.tracking_positions) * expansion_factor,
        )

        min_x_interp = interp1d(original_indices, min_x_vals, kind="linear")(
            interpolated_indices
        )
        min_y_interp = interp1d(original_indices, min_y_vals, kind="linear")(
            interpolated_indices
        )
        max_x_interp = interp1d(original_indices, max_x_vals, kind="linear")(
            interpolated_indices
        )
        max_y_interp = interp1d(original_indices, max_y_vals, kind="linear")(
            interpolated_indices
        )

        self.tracking_positions = [
            (int(min_x), int(min_y), int(max_x), int(max_y))
            for min_x, min_y, max_x, max_y in zip(
                min_x_interp, min_y_interp, max_x_interp, max_y_interp
            )
        ]

        print(f"{len(self.tracking_positions)=}")

    def create_cropped_animation(self, save_path, name, fade_duration=10):
        """Creates an animation of cropped frames with fade-in and fade-out effects."""
        fig, ax = plt.subplots(figsize=(9, 16))

        def update(frame_idx):
            ax.clear()
            frame = self.frames[frame_idx]
            min_x, min_y, max_x, max_y = self.tracking_positions[frame_idx]
            cropped_frame = frame[min_y:max_y, min_x:max_x]

            target_height = max_y - min_y
            target_width = int(target_height * 9 / 16)
            resized_cropped_frame = cv2.resize(
                cropped_frame,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )

            alpha = 1.0
            if frame_idx < fade_duration:
                alpha = frame_idx / fade_duration
            elif frame_idx > len(self.frames) - fade_duration:
                alpha = (len(self.frames) - frame_idx) / fade_duration

            ax.imshow(
                cv2.cvtColor(resized_cropped_frame, cv2.COLOR_BGR2RGB), alpha=alpha
            )
            ax.set_title(f"{name} (Frame {frame_idx})")
            ax.axis("off")

        print("Creating animation ...")
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.frames), interval=40, repeat=False
        )
        ani.save(save_path, writer="pillow", fps=25)
        print("Animation created")
        plt.close()

    def process_video(
        self, landmarks_data, save_path, name="Processed Video", fade_duration=10
    ):
        """Main method to process the video by cropping and saving the animation."""
        self.load_video_frames()
        crop_width, crop_height = self.calculate_fixed_crop_area(landmarks_data)
        self.get_tracking_positions(landmarks_data, crop_width, crop_height)
        self.smooth_crop_positions()
        self.create_cropped_animation(save_path, name, fade_duration)
