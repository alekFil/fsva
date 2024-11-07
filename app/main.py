import asyncio
import hashlib
import os
import pickle
import shutil
import tempfile

import gradio as gr
from utils.landmarks_processor import LandmarksProcessor
from utils.reels_processor import ReelsProcessor

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CACHE_DIR = "landmarks_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def generate_video_hash(video_path, step, model_path):
    """Генерирует хеш на основе содержимого видеофайла, параметра step и модели."""
    hash_md5 = hashlib.md5()

    # Включаем содержимое файла в хэш
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    # Включаем параметр step и путь к модели в хэш
    hash_md5.update(str(step).encode())
    hash_md5.update(model_path.encode())

    return hash_md5.hexdigest()


def load_cached_landmarks(video_hash):
    """Загружает landmarks_data из кэша, если оно существует."""
    cache_file = os.path.join(CACHE_DIR, f"{video_hash}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_cached_landmarks(video_hash, landmarks_data):
    """Сохраняет landmarks_data в кэш."""
    cache_file = os.path.join(CACHE_DIR, f"{video_hash}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(landmarks_data, f)


def process_video_inference(
    video_file,
    start_frame,
    end_frame,
    padding,
    draw_mode,
    step,
    model_choice,
):
    # Проверяем наличие загруженных файлов
    if video_file is None:
        return "Пожалуйста, загрузите видео."

    # Копируем видео во временное место и удаляем исходное
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    shutil.copyfile(video_file, temp_video_path)
    # os.remove(video_file)  # Удаляем нативный файл Gradio

    # Определяем путь к модели на основе выбора
    model_paths = {
        "Lite": "app/models/pose_landmarker_lite.task",
        "Full": "app/models/pose_landmarker_full.task",
        "Heavy": "app/models/pose_landmarker_heavy.task",
    }
    model_path = model_paths[model_choice]

    # Генерируем хеш видеофайла
    video_hash = generate_video_hash(temp_video_path, step, model_path)

    # Проверяем кэш
    landmarks_data = load_cached_landmarks(video_hash)
    if landmarks_data is None:
        # Если данных нет в кэше, запускаем процесс и сохраняем результат
        landmarks_data, world_landmarks_data, figure_masks_data = LandmarksProcessor(
            model_path,
            "0",
        ).process_video(temp_video_path, 25, step=step)

        # Сохраняем landmarks_data в кэш
        save_cached_landmarks(video_hash, landmarks_data)
    else:
        print("Данные landmarks загружены из кэша.")

    print(f"{landmarks_data.shape=}")

    # Проверяем диапазон кадров и конвертируем его
    try:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if start_frame >= end_frame:
            raise ValueError("Начальный кадр должен быть меньше конечного")
    except ValueError as e:
        # Возвращаем ошибку, если диапазон некорректен
        return str(e)

    reels_processor = ReelsProcessor(temp_video_path, video_fps=24, step=step)
    processed_video = reels_processor.process_jumps(
        [(start_frame, end_frame)],
        landmarks_data,
        padding=padding,
        draw_mode=draw_mode,
    )

    print("Обработанное видео сохранено как:", processed_video)

    return processed_video


def enable_button(video_file):
    # Включаем кнопку только если оба файла загружены
    return gr.update(interactive=bool(video_file))


if __name__ == "__main__":
    # Gradio Interface
    with gr.Blocks() as fsva:
        gr.Markdown(
            "## Обработка видео для создания одной анимации по заданному диапазону кадров"
        )

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(
                    label="Загрузите видео для обработки",
                    # format="mp4",
                    # autoplay=True,
                )

                start_frame = gr.Number(label="Начальный кадр", value=570)
                end_frame = gr.Number(label="Конечный кадр", value=630)
                padding = gr.Number(label="Padding (отступ для кропа)", value=0)

                # Поле для параметра step
                step = gr.Number(
                    label="Step (шаг пропуска кадров)",
                    value=1,
                    minimum=1,
                )

                # Переключатель выбора модели
                model_choice = gr.Radio(
                    label="Выберите модель",
                    choices=["Lite", "Full", "Heavy"],
                    value="Heavy",
                )

                # Переключатель режима отрисовки
                draw_mode = gr.Radio(
                    label="Выберите режим отрисовки",
                    choices=["Skeleton", "Trajectory", "Без отрисовки"],
                    value="Без отрисовки",
                )

                # Кнопка запуска
                run_button = gr.Button("Запустить обработку", interactive=False)
                # Отключаем кнопку, пока оба файла не будут загружены
                video_input.change(
                    enable_button,
                    [
                        video_input,
                    ],
                    run_button,
                )

            with gr.Column():
                video_output = gr.Video(
                    label="Обработанное видео",
                    width=360,
                    height=640,
                    autoplay=True,
                    loop=True,
                )

        # Настраиваем кнопку запуска, чтобы она выводила GIF в соседний столбец
        run_button.click(
            process_video_inference,
            inputs=[
                video_input,
                start_frame,
                end_frame,
                padding,
                draw_mode,
                step,
                model_choice,
            ],
            outputs=video_output,
        )

    fsva.launch(server_name="127.0.0.1", server_port=5548)
