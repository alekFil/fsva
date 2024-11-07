import asyncio
import shutil
import signal
import sys
import tempfile

import gradio as gr
from utils.animation_creator import AnimationCreator
from utils.landarks_converter import LandmarksConverter

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def process_video_inference(video_file, json_file, start_frame, end_frame):
    # Проверяем наличие загруженных файлов
    if video_file is None or json_file is None:
        return "Пожалуйста, загрузите видео и JSON файл."

    # Копируем видео во временное место и удаляем исходное
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    shutil.copyfile(video_file, temp_video_path)
    # os.remove(video_file)  # Удаляем нативный файл Gradio

    # Загружаем landmarks_data из JSON файла, используя путь к файлу
    with open(json_file.name, "r") as f:
        landmarks_data = f.read()
    landmarks_tensor = LandmarksConverter.json_to_tensor(landmarks_data)

    print(f"{landmarks_tensor.shape=}")

    # Проверяем диапазон кадров и конвертируем его
    try:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if start_frame >= end_frame:
            raise ValueError("Начальный кадр должен быть меньше конечного")
    except ValueError as e:
        return str(e)  # Возвращаем ошибку, если диапазон некорректен

    processor = AnimationCreator(
        video_path=temp_video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        image_shape=(1920, 1080),
    )
    processor.process_video(landmarks_tensor, "output1.gif")

    # reels_processor = ReelsProcessor(temp_video_path, video_fps=24)
    # processed_video = reels_processor.process_jumps([(start_frame, end_frame)])

    # print("Обработанное видео сохранено как:", processed_video)

    # Задаем фиктивный выход, пока анимация не реализована
    output_gifs = "output1.gif"

    return output_gifs


def enable_button(video_file, json_file):
    # Включаем кнопку только если оба файла загружены
    return gr.update(interactive=bool(video_file and json_file))


def signal_handler(sig, frame):
    print("Завершение работы Gradio...")
    sys.exit(0)


if __name__ == "__main__":
    # Устанавливаем обработчик сигналов для корректного завершения
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
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
                json_input = gr.File(
                    label="Загрузите landmarks_data (JSON)",
                    file_types=[".json"],
                )
                start_frame = gr.Number(label="Начальный кадр", value=190)
                end_frame = gr.Number(label="Конечный кадр", value=210)

                # Кнопка запуска
                run_button = gr.Button("Запустить обработку", interactive=False)
                # Отключаем кнопку, пока оба файла не будут загружены
                video_input.change(enable_button, [video_input, json_input], run_button)
                json_input.change(enable_button, [video_input, json_input], run_button)

            with gr.Column():
                gif_output = gr.Image(label="GIF", height=640, width=360)

        # Настраиваем кнопку запуска, чтобы она выводила GIF в соседний столбец
        run_button.click(
            process_video_inference,
            inputs=[video_input, json_input, start_frame, end_frame],
            outputs=gif_output,
        )

    fsva.launch(server_name="127.0.0.1", server_port=5548)
