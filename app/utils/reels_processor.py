import os
import subprocess
import tempfile

import cv2


class ReelsProcessor:
    def __init__(self, input_video, video_fps=25):
        """
        Инициализация процессора видео.

        Параметры:
        - input_video: Путь к исходному видео.
        - video_fps: Частота кадров видео (по умолчанию 25).
        """
        self.input_video = input_video
        self.video_fps = video_fps
        self.temp_files = []
        self.temp_dir = tempfile.mkdtemp()  # Временная директория для кадров

    def draw_trajectory(
        self, frame, center_points, point_color=(0, 0, 255), line_color=(0, 255, 0)
    ):
        """
        Рисует центральную точку и траекторию на кадре.

        Параметры:
        - frame: текущий кадр видео.
        - center_points: список координат центральной точки на каждом кадре.
        - point_color: цвет центральной точки в формате BGR (по умолчанию красный).
        - line_color: цвет линии траектории в формате BGR (по умолчанию зеленый).

        Возвращает:
        - frame: кадр с нарисованной траекторией.
        """
        for i, point in enumerate(center_points):
            cv2.circle(frame, point, 5, point_color, -1)  # Рисуем центральную точку
            if i > 0:
                cv2.line(
                    frame, center_points[i - 1], point, line_color, 2
                )  # Рисуем линию траектории
        return frame

    def process_jumps(self, jump_frames, landmarks_tensor):
        """
        Обрабатывает видео, нарезая фрагменты с прыжками, добавляя центральную точку и её траекторию.

        Параметры:
        - jump_frames: Список кортежей с начальным и конечным кадрами для каждого прыжка.
        - landmarks_tensor: Тензор с координатами точек скелета.

        Возвращает:
        - Путь к финальному видео с обработанными фрагментами.
        """
        cap = cv2.VideoCapture(self.input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        center_points = []
        hand_points = []
        frame_count = 0

        # Обрабатываем кадры и сохраняем их в виде изображений
        for start_frame, end_frame in jump_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                # Получаем координаты центральной точки из landmarks_tensor
                center_x = int(
                    (
                        landmarks_tensor[frame_idx // 3, 23, 0]
                        + landmarks_tensor[frame_idx // 3, 24, 0]
                    )
                    / 2
                    * width
                )
                center_y = int(
                    (
                        landmarks_tensor[frame_idx // 3, 23, 1]
                        + landmarks_tensor[frame_idx // 3, 24, 1]
                    )
                    / 2
                    * height
                )
                center_points.append((center_x, center_y))

                frame = self.draw_trajectory(
                    frame,
                    center_points,
                    point_color=(102, 153, 0),
                    line_color=(102, 153, 0),
                )

                # Получаем координаты центральной точки из landmarks_tensor
                hand_x = int(landmarks_tensor[frame_idx // 3, 0, 0] * width)
                hand_y = int(landmarks_tensor[frame_idx // 3, 0, 1] * height)
                hand_points.append((hand_x, hand_y))

                # Рисуем центральную точку и траекторию
                frame = self.draw_trajectory(
                    frame,
                    hand_points,
                    point_color=(0, 102, 153),
                    line_color=(0, 102, 153),
                )

                # Сохраняем кадр в виде изображения
                frame_path = os.path.join(self.temp_dir, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_path, frame)
                frame_count += 1

        cap.release()

        # Создание видео из кадров с помощью ffmpeg
        output_video_path = "processed_video_with_trajectory.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(self.video_fps),
                "-i",
                os.path.join(self.temp_dir, "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                output_video_path,
            ]
        )

        # Очистка временных файлов
        self.cleanup_temp_files()

        return output_video_path

    def cut_video_segment(self, start_time, end_time, output_path):
        command = [
            "ffmpeg",
            "-y",
            "-i",
            self.input_video,
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            output_path,
        ]
        subprocess.run(command, check=True)
        self.temp_files.append(output_path)

    def apply_fade_effect(
        self, input_path, output_path, fade_in_duration=0.5, fade_out_duration=0.5
    ):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_path,
            ],
            capture_output=True,
            text=True,
        )
        duration = float(probe.stdout.strip())

        fade_in_duration = min(0.2 * duration, 0.5)
        fade_out_duration = min(0.2 * duration, 0.5)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            f"fade=t=in:st=0:d={fade_in_duration},fade=t=out:st={duration - fade_out_duration}:d={fade_out_duration}",
            "-c:a",
            "copy",
            output_path,
        ]
        subprocess.run(command, check=True)
        self.temp_files.append(output_path)

    def concat_clips(self, clips, output_path):
        with open("filelist.txt", "w") as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "filelist.txt",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            output_path,
        ]
        subprocess.run(command, check=True)

        os.remove("filelist.txt")

    def cleanup_temp_files(self):
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)
        self.temp_files.clear()
