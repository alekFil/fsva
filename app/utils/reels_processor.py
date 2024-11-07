import os
import subprocess
import tempfile
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm


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
        self.skeleton_connections = [
            (0, 2),
            (0, 5),
            (2, 7),
            (5, 8),
            (5, 4),
            (5, 6),
            (2, 1),
            (2, 3),
            (10, 9),
            (11, 12),
            (12, 14),
            (14, 16),
            (16, 22),
            (16, 20),
            (20, 18),
            (18, 16),
            (11, 13),
            (13, 15),
            (15, 21),
            (15, 19),
            (19, 17),
            (17, 15),
            (12, 24),
            (11, 23),
            (23, 24),
            (24, 26),
            (26, 28),
            (28, 30),
            (30, 32),
            (32, 28),
            (23, 25),
            (25, 27),
            (27, 29),
            (29, 31),
            (31, 27),
        ]

    def interpolate_landmarks(self, landmarks_tensor):
        """
        Интерполирует нулевые значения в landmarks_tensor, усредняя по соседним значениям.
        """
        num_frames, num_points, num_coords = landmarks_tensor.shape

        for frame_idx in tqdm(range(num_frames)):
            for point_idx in range(num_points):
                # Проверяем, если все три координаты точки равны 0
                if np.array_equal(landmarks_tensor[frame_idx, point_idx], [0, 0, 0]):
                    prev_frame_idx = frame_idx - 1
                    next_frame_idx = frame_idx + 1

                    # Найти ближайшие предыдущий и следующий кадры с ненулевыми координатами для этой точки
                    while prev_frame_idx >= 0 and np.array_equal(
                        landmarks_tensor[prev_frame_idx, point_idx], [0, 0, 0]
                    ):
                        prev_frame_idx -= 1
                    while next_frame_idx < num_frames and np.array_equal(
                        landmarks_tensor[next_frame_idx, point_idx], [0, 0, 0]
                    ):
                        next_frame_idx += 1

                    # Если найдены валидные предыдущий и следующий кадры, усредняем
                    if prev_frame_idx >= 0 and next_frame_idx < num_frames:
                        landmarks_tensor[frame_idx, point_idx] = (
                            landmarks_tensor[prev_frame_idx, point_idx]
                            + landmarks_tensor[next_frame_idx, point_idx]
                        ) / 2
                    elif (
                        prev_frame_idx >= 0
                    ):  # Если есть только предыдущий, используем его координаты
                        landmarks_tensor[frame_idx, point_idx] = landmarks_tensor[
                            prev_frame_idx, point_idx
                        ]
                    elif (
                        next_frame_idx < num_frames
                    ):  # Если есть только следующий, используем его координаты
                        landmarks_tensor[frame_idx, point_idx] = landmarks_tensor[
                            next_frame_idx, point_idx
                        ]

        return landmarks_tensor

    def draw_trajectory(
        self,
        frame,
        center_points,
        point_color=(0, 0, 255),
        line_color=(0, 255, 0),
        interpolation_factor=1,
    ):
        """
        Рисует центральную точку и сглаженную траекторию на кадре.

        Параметры:
        - frame: текущий кадр видео.
        - center_points: список координат центральной точки на каждом кадре.
        - point_color: цвет центральной точки в формате BGR (по умолчанию красный).
        - line_color: цвет линии траектории в формате BGR (по умолчанию зеленый).
        - interpolation_factor: фактор интерполяции для создания дополнительных точек.

        Возвращает:
        - frame: кадр с нарисованной траекторией.
        """
        if len(center_points) > 1:
            # Разделение координат x и y для интерполяции
            x_points = [p[0] for p in center_points]
            y_points = [p[1] for p in center_points]

            # Создаем параметрическое представление данных для интерполяции
            t = np.arange(len(center_points))
            t_interpolated = np.linspace(
                0, len(center_points) - 1, len(center_points) * interpolation_factor
            )

            # Интерполяция координат x и y
            x_smooth = np.interp(t_interpolated, t, x_points)
            y_smooth = np.interp(t_interpolated, t, y_points)

            # Формируем сглаженные точки в формате, подходящем для cv2.polylines
            smooth_points = np.array(
                [[[int(x), int(y)] for x, y in zip(x_smooth, y_smooth)]], dtype=np.int32
            )

            # Рисуем сглаженную траекторию
            cv2.polylines(
                frame, smooth_points, isClosed=False, color=line_color, thickness=2
            )

        # Рисуем центральные точки для наглядности
        for point in center_points:
            cv2.circle(frame, point, 5, point_color, -1)  # Рисуем центральную точку

        return frame

    def draw_skeleton(
        self,
        frame,
        joints,
        point_color=(0, 255, 0),
        line_color=(255, 0, 0),
    ):
        """
        Рисует скелет, соединяя точки суставов в кадре.

        Параметры:
        - frame: текущий кадр видео.
        - joints: тензор координат суставов для одного кадра, размер (33, 3), где joints[i] = (x, y, z) для сустава i.
        - connections: список кортежей, указывающих, какие суставы нужно соединить.
        - point_color: цвет точек суставов в формате BGR.
        - line_color: цвет линий, соединяющих суставы, в формате BGR.

        Возвращает:
        - frame: кадр с нарисованным скелетом.
        """
        height, width = frame.shape[:2]  # Получаем размеры кадра

        print(f"{joints=}")
        # Преобразуем тензор в формат NumPy, оставляем только координаты x и y и масштабируем их
        joints_np = joints[:, :2].astype(int)
        print(f"{joints_np=}")

        # Отрисовка соединений между суставами
        for start_idx, end_idx in self.skeleton_connections:
            if start_idx < len(joints_np) and end_idx < len(joints_np):
                start_point = tuple(joints_np[start_idx])
                end_point = tuple(joints_np[end_idx])
                # Рисуем линию между суставами
                cv2.line(frame, start_point, end_point, line_color, 2)

        # Отрисовка точек суставов
        for joint in joints_np:
            cv2.circle(frame, tuple(joint), 5, point_color, -1)

        return frame

    def process_jumps(
        self,
        jump_frames,
        landmarks_tensor,
        smooth_window=10,
        padding=200,
        draw_mode="Trajectory",
    ):
        # Применяем интерполяцию для замены нулевых значений в landmarks_tensor
        landmarks_tensor = self.interpolate_landmarks(landmarks_tensor)

        cap = cv2.VideoCapture(self.input_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Устанавливаем размеры для кропа с соотношением 9:16 и делаем их четными
        crop_width = min(width, int(height * (9 / 16))) + 2 * padding
        crop_height = min(height, int(width * (16 / 9))) + 2 * padding

        # Округляем до ближайших четных чисел, чтобы соответствовать требованиям кодека
        crop_width = crop_width if crop_width % 2 == 0 else crop_width - 1
        crop_height = crop_height if crop_height % 2 == 0 else crop_height - 1

        center_points = []
        hand_points = []
        frame_count = 0

        # Дек для хранения последних координат центральной точки для сглаживания
        recent_centers = deque(maxlen=smooth_window)

        for start_frame, end_frame in jump_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in tqdm(range(start_frame, end_frame + 1)):
                ret, frame = cap.read()
                if not ret:
                    break

                # Получаем исходные координаты центральной точки из landmarks_tensor
                original_center_x = int(
                    (
                        landmarks_tensor[frame_idx // 3, 23, 0]
                        + landmarks_tensor[frame_idx // 3, 24, 0]
                    )
                    / 2
                    * width
                )
                original_center_y = int(
                    (
                        landmarks_tensor[frame_idx // 3, 23, 1]
                        + landmarks_tensor[frame_idx // 3, 24, 1]
                    )
                    / 2
                    * height
                )
                center_points.append((original_center_x, original_center_y))

                # Добавляем координаты центральной точки в очередь для сглаживания
                recent_centers.append((original_center_x, original_center_y))

                # Вычисляем усредненные координаты центра для сглаживания
                avg_center_x = int(np.mean([pt[0] for pt in recent_centers]))
                avg_center_y = int(np.mean([pt[1] for pt in recent_centers]))

                # Рассчитываем координаты для обрезки вокруг усредненного центра
                x1 = max(0, min(avg_center_x - crop_width // 2, width - crop_width))
                y1 = max(0, min(avg_center_y - crop_height // 2, height - crop_height))
                x2 = x1 + crop_width
                y2 = y1 + crop_height

                # Выполняем кроп кадра
                cropped_frame = frame[y1:y2, x1:x2]

                # Обновляем список центральных точек для кропнутого кадра
                cropped_center_points = [(x - x1, y - y1) for x, y in center_points]

                # Получаем исходные координаты руки и корректируем их относительно кропа
                original_hand_x = int(landmarks_tensor[frame_idx // 3, 0, 0] * width)
                original_hand_y = int(landmarks_tensor[frame_idx // 3, 0, 1] * height)
                hand_points.append((original_hand_x, original_hand_y))

                # Обновляем список точек руки для кропнутого кадра
                cropped_hand_points = [(x - x1, y - y1) for x, y in hand_points]

                if draw_mode == "Trajectory":
                    # Рисуем центральную точку и траекторию на кропнутом кадре
                    cropped_frame = self.draw_trajectory(
                        cropped_frame,
                        cropped_center_points,
                        point_color=(102, 153, 0),
                        line_color=(102, 153, 0),
                    )

                    # Рисуем траекторию руки на кропнутом кадре
                    cropped_frame = self.draw_trajectory(
                        cropped_frame,
                        cropped_hand_points,
                        point_color=(0, 102, 153),
                        line_color=(0, 102, 153),
                    )
                elif draw_mode == "Skeleton":
                    # Пересчитываем координаты суставов для кропнутого кадра
                    joints = (
                        landmarks_tensor[frame_idx // 3, :, :2]
                        * np.array([width, height])
                    ).numpy()
                    cropped_joints = np.array(
                        [(int(x - x1), int(y - y1)) for x, y in joints]
                    )
                    cropped_frame = self.draw_skeleton(cropped_frame, cropped_joints)

                # Сохраняем кропнутый кадр в виде изображения
                frame_path = os.path.join(self.temp_dir, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_path, cropped_frame)
                frame_count += 1

        cap.release()

        # Создание видео из кропнутых кадров с помощью ffmpeg
        output_video_path = "processed_video_with_crop.mp4"
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

        # Применяем fade-in и fade-out к финальному видео
        faded_output_path = "processed_video_with_fade.mp4"
        self.apply_fade_effect(
            output_video_path,
            faded_output_path,
            fade_in_duration=1,
            fade_out_duration=1,
        )

        # Очистка временных файлов
        self.cleanup_temp_files()

        return faded_output_path

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
        self,
        input_path,
        output_path,
        fade_in_duration=0.5,
        fade_out_duration=0.5,
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
