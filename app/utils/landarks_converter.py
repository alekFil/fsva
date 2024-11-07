import json

import torch


class LandmarksConverter:
    @staticmethod
    def tensor_to_json(landmarks_tensor):
        """
        Преобразует тензор landmarks_data в JSON-структуру.
        :param landmarks_tensor: torch.Tensor с размером [frames, points, 3]
        :return: JSON-строка, представляющая landmarks_data
        """
        # Конвертируем тензор в список, который можно сохранить в JSON
        landmarks_list = landmarks_tensor.tolist()
        landmarks_json = {"landmarks_data": landmarks_list}
        return landmarks_json

    @staticmethod
    def json_to_tensor(landmarks_json):
        """
        Преобразует JSON-структуру landmarks_data обратно в тензор.
        :param landmarks_json: JSON-строка с данными landmarks_data
        :return: torch.Tensor с размером [frames, points, 3]
        """
        # Загружаем JSON-данные и конвертируем их в тензор
        landmarks_dict = json.loads(landmarks_json)
        landmarks_list = landmarks_dict["landmarks_data"]
        landmarks_tensor = torch.tensor(landmarks_list, dtype=torch.float16)
        return landmarks_tensor

    @staticmethod
    def save_to_file(landmarks_tensor, file_path):
        """
        Сохраняет тензор landmarks_data в файл в формате JSON.
        :param landmarks_tensor: torch.Tensor с размером [frames, points, 3]
        :param file_path: путь к файлу для сохранения
        """
        landmarks_json = LandmarksConverter.tensor_to_json(landmarks_tensor)
        with open(file_path, "w") as f:
            json.dump(landmarks_json, f, indent=4)
        print(f"Данные сохранены в {file_path}")

    @staticmethod
    def load_from_file(file_path):
        """
        Загружает JSON-данные landmarks_data из файла и преобразует их в тензор.
        :param file_path: путь к JSON-файлу
        :return: torch.Tensor с размером [frames, points, 3]
        """
        with open(file_path, "r") as f:
            landmarks_json = json.load(f)
        return LandmarksConverter.json_to_tensor(landmarks_json)


if __name__ == "__main__":
    # Создаем пример тензора для тестирования
    landmarks_tensor = torch.rand((2, 33, 3), dtype=torch.float32)

    # Преобразование тензора в JSON
    landmarks_json = LandmarksConverter.tensor_to_json(landmarks_tensor)
    print("JSON Data:", landmarks_json)

    # Преобразование JSON обратно в тензор
    converted_tensor = LandmarksConverter.json_to_tensor(landmarks_json)
    print("Tensor:", converted_tensor)
