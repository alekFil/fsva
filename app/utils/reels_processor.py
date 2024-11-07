import os
import subprocess


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

    def cut_video_segment(self, start_time, end_time, output_path):
        """
        Вырезает сегмент видео.

        Параметры:
        - start_time: Время начала сегмента (в секундах).
        - end_time: Время окончания сегмента (в секундах).
        - output_path: Путь для сохранения вырезанного фрагмента.
        """
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
        """
        Добавляет эффекты плавного появления и затухания к видеофрагменту.
        Длительность затухания автоматически адаптируется к длине видеофрагмента.

        Параметры:
        - input_path: Путь к исходному видеофрагменту.
        - output_path: Путь для сохранения обработанного видео.
        """
        # Определяем длительность фрагмента
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

        # Устанавливаем длительность появления и затухания как 20% от длины фрагмента, но не больше 0.5 сек
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

    def process_jumps(self, jump_frames):
        """
        Обрабатывает видео, нарезая фрагменты с прыжками и добавляя переходы.

        Параметры:
        - jump_frames: Список кортежей с начальным и конечным кадрами для каждого прыжка.

        Возвращает:
        - Путь к финальному видео с обработанными фрагментами.
        """
        processed_clips = []

        # Обрабатываем каждый фрагмент
        for i, (start_frame, end_frame) in enumerate(jump_frames):
            start_time = start_frame / self.video_fps
            end_time = end_frame / self.video_fps
            temp_clip_path = f"temp_clip_{i}.mp4"
            faded_clip_path = f"faded_clip_{i}.mp4"

            # Вырезаем сегмент
            self.cut_video_segment(start_time, end_time, temp_clip_path)

            # Применяем эффект перехода
            self.apply_fade_effect(temp_clip_path, faded_clip_path)
            processed_clips.append(faded_clip_path)

        # Объединяем все фрагменты в финальное видео
        final_output = "final_output.mp4"
        self.concat_clips(processed_clips, final_output)

        # Очищаем временные файлы
        self.cleanup_temp_files()

        return final_output

    def concat_clips(self, clips, output_path):
        """
        Объединяет видеофрагменты в один файл.

        Параметры:
        - clips: Список путей к видеофрагментам для объединения.
        - output_path: Путь для сохранения объединенного видео.
        """
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

        # Удаляем временный файл списка
        os.remove("filelist.txt")

    def cleanup_temp_files(self):
        """
        Удаляет временные файлы, созданные во время обработки.
        """
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        self.temp_files.clear()
