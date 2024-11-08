import asyncio
import hashlib
import os
import pickle
import shutil
import tempfile

import gradio as gr
import numpy as np
import torch
from utils.inferences.inference_elements import ModelInferenceService
from utils.landmarks_processor import LandmarksProcessor
from utils.reels_processor import ReelsProcessor

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CACHE_DIR = "landmarks_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Загрузка модели при старте сервиса
model_path = "app\models\elements\checkpoints\checkpoint.pt"
parameters = (64, 2, 0.3, 198, 0.05, 128, True, True, True, 0.05)
num_classes = 3
inference_service = ModelInferenceService(model_path, parameters, num_classes)


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
    cache_file = os.path.join(CACHE_DIR, f"{video_hash}_landmarks.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            landmarks_data = pickle.load(f)
    """Загружает world_landmarks_data из кэша, если оно существует."""
    cache_file = os.path.join(CACHE_DIR, f"{video_hash}_world_landmarks.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            world_landmarks_data = pickle.load(f)
            return landmarks_data, world_landmarks_data
    return None, None


def save_cached_landmarks(video_hash, landmarks_data, world_landmarks_data):
    """Сохраняет landmarks_data в кэш."""
    cache_file = os.path.join(CACHE_DIR, f"{video_hash}_landmarks.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(landmarks_data, f)
    """Сохраняет world_landmarks_data в кэш."""
    cache_file = os.path.join(CACHE_DIR, f"{video_hash}_world_landmarks.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(world_landmarks_data, f)


def predict(landmarks_data, world_landmarks_data):
    # Преобразование входных данных в тензор
    check_data = torch.load(
        "app/models/elements/checkpoints/check_v38.pt", weights_only=False
    )

    landmarks_tensor = torch.tensor(landmarks_data)
    world_landmarks_tensor = torch.tensor(world_landmarks_data)
    print(f"{landmarks_tensor.shape=}")
    print(f"{world_landmarks_tensor.shape=}")
    features = torch.cat([world_landmarks_tensor, landmarks_tensor], dim=2)
    features = features.view(features.size(0), -1)  # Shape: [1841, 198]
    print(f"{features.shape=}")

    total_elements = features.size(0)
    sequence_length = 25
    num_full_sequences = total_elements // sequence_length
    last_sequence_length = total_elements % sequence_length
    total_sequences = (
        num_full_sequences + 1 if last_sequence_length > 0 else num_full_sequences
    )

    features_prepared = torch.zeros(
        total_sequences, sequence_length, features.size(1)
    )  # Shape: [88, 25, 198]

    for i in range(num_full_sequences):
        features_prepared[i] = features[i * sequence_length : (i + 1) * sequence_length]
    if last_sequence_length > 0:
        last_sequence_data = features[num_full_sequences * sequence_length :]
        features_prepared[-1, :last_sequence_length] = last_sequence_data

    lengths = [sequence_length] * num_full_sequences
    if last_sequence_length > 0:
        lengths.append(last_sequence_length)
    lengths = torch.tensor(np.array(lengths))

    print(f"{features_prepared.shape=}")
    print(f"{lengths.shape=}")
    print(f"{lengths=}")

    labels_batch, lengths_batch, validation_mask_batch, swfeatures_batch = check_data
    # lengths_batch, swfeatures_batch = lengths.clone(), features_prepared.clone()
    print(f"{lengths_batch.shape=}")
    print(f"{swfeatures_batch.shape=}")
    print(f"{swfeatures_batch[0]=}")

    max_len = lengths_batch.max()
    mask = torch.arange(max_len).expand(
        len(lengths_batch), max_len
    ) < lengths_batch.unsqueeze(1)

    predicted_labels, predicted_probs = inference_service.predict(
        features=swfeatures_batch,
        lengths=lengths_batch,
        mask=mask,
    )

    return predicted_labels, predicted_probs


def process_video_inference(
    video_file,
    frame_ranges_str,
    padding,
    draw_mode,
    step,
    model_choice,
    progress=gr.Progress(track_tqdm=True),
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
        "Lite": "app/models/landmarkers/pose_landmarker_lite.task",
        "Full": "app/models/landmarkers/pose_landmarker_full.task",
        "Heavy": "app/models/landmarkers/pose_landmarker_heavy.task",
    }
    model_path = model_paths[model_choice]

    # Генерируем хеш видеофайла
    video_hash = generate_video_hash(temp_video_path, step, model_path)

    # Проверяем кэш
    landmarks_data, world_landmarks_data = load_cached_landmarks(video_hash)
    if landmarks_data is None:
        # Если данных нет в кэше, запускаем процесс и сохраняем результат
        landmarks_data, world_landmarks_data, figure_masks_data = LandmarksProcessor(
            model_path,
            "0",
        ).process_video(temp_video_path, step=step)

        # Сохраняем landmarks_data в кэш
        save_cached_landmarks(video_hash, landmarks_data, world_landmarks_data)
    else:
        print("Данные landmarks загружены из кэша.")

    # Обрабатываем строку с диапазонами кадров
    try:
        # Преобразуем строку в список кортежей
        frame_ranges = [
            tuple(map(int, item.strip().replace("(", "").replace(")", "").split(",")))
            for item in frame_ranges_str.split("),")
        ]
        # Проверка, что начальный кадр меньше конечного в каждом диапазоне
        for start, end in frame_ranges:
            if start >= end:
                raise ValueError(
                    f"Начальный кадр должен быть меньше конечного: ({start}, {end})"
                )
    except ValueError as e:
        return f"Ошибка в формате диапазонов: {str(e)}"

    # Здесь нужно получить фрагменты из предсказательной модели
    predicted_labels, _ = predict(landmarks_data, world_landmarks_data)
    print(f"{predicted_labels[405:420]=}")
    # print(predicted_labels)

    return "processed_video_with_fades.mp4"

    reels_processor = ReelsProcessor(temp_video_path, step=step)
    processed_video = reels_processor.process_jumps(
        frame_ranges,
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

                # Текстовое поле для диапазонов кадров
                frame_ranges_str = gr.Textbox(
                    label="Диапазоны кадров",
                    placeholder="Введите диапазоны в формате (400, 500), (600, 700)",
                )

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
                frame_ranges_str,
                padding,
                draw_mode,
                step,
                model_choice,
            ],
            outputs=video_output,
        )

    fsva.launch(server_name="127.0.0.1", server_port=5548)
