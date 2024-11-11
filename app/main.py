import asyncio
import hashlib
import os
import pickle
import shutil
import tempfile

import gradio as gr
import torch
from torch.nn.utils.rnn import pad_sequence
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


def find_reels_fragments(labels, target_class, batch_size):
    fragments = []

    # Параметры для поиска последовательностей
    start = None
    count = 0

    for i, label in enumerate(labels):
        if label == target_class:
            if start is None:
                start = i
            count += 1
        else:
            if start is not None and count >= 1:
                # Определяем индекс среднего элемента
                middle_index = start + count // 2

                # Определяем, к какому батчу относится средний элемент
                batch_index = middle_index // batch_size

                # Определяем начало и конец соседних батчей
                start_batch = max(0, (batch_index - 1) * batch_size)
                end_batch = min(len(labels) - 1, (batch_index + 2) * batch_size - 1)

                # Объединяем с предыдущим фрагментом, если они пересекаются
                if fragments and start_batch <= fragments[-1][1]:
                    # Обновляем конец последнего фрагмента
                    fragments[-1] = (fragments[-1][0], max(fragments[-1][1], end_batch))
                else:
                    # Добавляем новый фрагмент
                    fragments.append((start_batch, end_batch))

            # Сброс параметров
            start = None
            count = 0

    # Проверка для последней последовательности
    if start is not None and count >= 3:
        middle_index = start + count // 2
        batch_index = middle_index // batch_size
        start_batch = max(0, (batch_index - 1) * batch_size)
        end_batch = min(len(labels) - 1, (batch_index + 2) * batch_size - 1)

        # Объединяем с предыдущим фрагментом, если они пересекаются
        if fragments and start_batch <= fragments[-1][1]:
            fragments[-1] = (fragments[-1][0], max(fragments[-1][1], end_batch))
        else:
            fragments.append((start_batch, end_batch))

    return fragments


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
    landmarks_tensor = torch.tensor(landmarks_data)
    world_landmarks_tensor = torch.tensor(world_landmarks_data)
    print(f"{landmarks_tensor.shape=}")
    print(f"{world_landmarks_tensor.shape=}")

    def collate_ml(batch):
        (
            lengths,
            swfeatures,
            sfeatures,
        ) = zip(*batch)

        lengths = torch.tensor(lengths).flatten()

        swfeatures = pad_sequence(swfeatures, batch_first=True)
        swfeatures = swfeatures.view(swfeatures.shape[0], swfeatures.shape[1], -1)

        sfeatures = pad_sequence(sfeatures, batch_first=True)
        sfeatures = sfeatures.view(sfeatures.shape[0], sfeatures.shape[1], -1)

        features = torch.cat(
            [
                swfeatures,
                sfeatures,
            ],
            dim=2,
        )

        return lengths, features

    # Определим длину каждой последовательности и необходимое количество батчей
    sequence_length = 25
    num_sequences = (
        landmarks_tensor.shape[0] + sequence_length - 1
    ) // sequence_length  # Округление вверх

    # Разделим данные на последовательности по 25 элементов
    sequences = []

    for i in range(num_sequences):
        start_idx = i * sequence_length
        end_idx = min(start_idx + sequence_length, landmarks_tensor.shape[0])

        # Получаем последовательность и вычисляем её истинную длину
        seq_landmarks = landmarks_tensor[start_idx:end_idx]
        seq_world_landmarks = world_landmarks_tensor[start_idx:end_idx]
        true_length = seq_landmarks.shape[0]

        # Дополняем последовательности нулями до длины 25, если они короче
        if true_length < sequence_length:
            padding = torch.zeros(sequence_length - true_length, 33, 3)
            seq_landmarks = torch.cat([seq_landmarks, padding], dim=0)
            seq_world_landmarks = torch.cat([seq_world_landmarks, padding], dim=0)

        # Добавляем последовательности и их длины в список
        sequences.append((true_length, seq_world_landmarks, seq_landmarks))

    lengths_tensor, features = collate_ml(sequences)

    print(f"{features.shape=}")
    print(f"{lengths_tensor.shape=}")
    print(f"{lengths_tensor=}")

    print(f"{features[0]=}")

    lengths_batch, swfeatures_batch = lengths_tensor.clone(), features.clone()
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

    # Здесь нужно получить фрагменты из предсказательной модели
    predicted_labels, _ = predict(landmarks_data, world_landmarks_data)
    print(f"{predicted_labels[405:420]=}")
    # print(predicted_labels)

    reels_fragments = find_reels_fragments(predicted_labels, 1, 25)
    print(reels_fragments)
    reels = [(x * 3, y * 3) for x, y in reels_fragments]

    reels_processor = ReelsProcessor(temp_video_path, step=step)
    processed_video = reels_processor.process_jumps(
        tuple(reels),
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

                padding = gr.Number(label="Padding (отступ для кропа)", value=0)

                # Поле для параметра step
                step = gr.Number(
                    label="Step (шаг пропуска кадров)",
                    value=3,
                    minimum=1,
                )

                # Переключатель выбора модели
                model_choice = gr.Radio(
                    label="Выберите модель",
                    choices=["Lite", "Full", "Heavy"],
                    value="Lite",
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
                padding,
                draw_mode,
                step,
                model_choice,
            ],
            outputs=video_output,
        )

    fsva.launch(server_name="127.0.0.1", server_port=5548)
