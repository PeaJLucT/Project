import torch
import librosa
import numpy as np
import torch.nn.functional as F
import pandas as pd
from CNN import MusicCNN
from dataset import Mel 
import random
class GenrePredictor:
    def __init__(self, model_path, input_shape, num_classes, genre_map):
        """
        Args:
            model_path (str): Путь к файлу с весами модели(.pth)
            input_shape (tuple): (каналы, высота, ширина)
            num_classes (int): Количество жанров
            genre_map (dict): {0: 'Hip-Hop', 1: 'Pop', ...}
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #Загружаем архитектуру
        self.model = MusicCNN(input_shape=input_shape, num_classes=num_classes)
        #Загружаем обученные веса
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        #Переводим модель в режим оценки
        self.model.eval()
        self.model.to(self.device)
        
        self.mel_generator = Mel()
        self.input_shape = input_shape
        self.genre_map = genre_map

    def predict_genres(self, audio_path, seconds_per_sample=5, threshold=0.1):
        """
        Предсказывает жанры для одного аудиофайла

        Args:
            audio_path (str): Путь к аудиофайлу (mp3, wav...)
            seconds_per_sample (int): Длительность сэмплов в секундах (5)
            threshold (float): Порог вероятности для включения жанра в итоговый список

        Returns:
            list: название_жанра, вероятность
        """
        try:
            # 1. Загружаем и нарезаем аудио на сэмплы
            duration = librosa.get_duration(path=audio_path)
            num_samples = int(duration // seconds_per_sample)
            
            if num_samples == 0:
                print(f"трек {audio_path} меньше {seconds_per_sample} секунд, предсказание невозможно")
                return []
            all_probabilities = []
            
            # проходим по каждому сэмплу
            for i in range(num_samples):
                offset = i * seconds_per_sample
                y, sr = librosa.load(audio_path, offset=offset, duration=seconds_per_sample)
                spectrogram = self.mel_generator.generate(y, sr)

                spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                spectrogram_tensor = spectrogram_tensor.to(self.device)
                
                #предсказание для одного сэмпла
                with torch.no_grad():
                    logits = self.model(spectrogram_tensor)
                    probabilities = torch.sigmoid(logits)
                    all_probabilities.append(probabilities.cpu().numpy())
            if not all_probabilities:
                return []
            # print(f'Жанры по вероятностям: {all_probabilities}')
            avg_probabilities = np.mean(all_probabilities, axis=0)[0]
            # print(f'avg_prob: {len(avg_probabilities)}')
            # итоговый список жанров
            result = []
            for i, prob in enumerate(avg_probabilities):
                if prob >= threshold:
                    genre_name = self.genre_map.get(i, f"Неизвестный жанр {i}")
                    result.append((genre_name, float(prob)))
            # Сортируем по убыванию

            result.sort(key=lambda x: x[1], reverse=True)
            return result
        except Exception as e:
            print(f"Произошла ошибка при обработке файла {audio_path}: {e}")
            return []
"""
# Функция для проверки жанров своей песни
if __name__ == '__main__':
    genres_df = pd.read_csv('final_dataset/track_genres.csv', index_col=0)
    genre_columns = genres_df.columns.astype(int).tolist() # Получаем ID жанров
    
    all_genres_info = pd.read_csv('Musics/fma_metadata/fma_metadata/genres.csv', index_col=0)
    genre_id_to_name = all_genres_info['title'].to_dict()
    
    # {индекс_колонки: название_жанра}
    final_genre_map = {i: genre_id_to_name[genre_id] for i, genre_id in enumerate(genre_columns)}

    SECONDS_PER_SAMPLE = 5
    SAMPLE_RATE = 22050
    mel_generator = Mel()
    dummy_audio = np.zeros(SAMPLE_RATE * SECONDS_PER_SAMPLE)
    dummy_spectrogram = mel_generator.generate(dummy_audio, SAMPLE_RATE)
    dummy_tensor = torch.tensor(dummy_spectrogram).unsqueeze(0)
    
    input_shape = dummy_tensor.shape
    num_classes = len(genres_df.columns)
    # print(f'{input_shape}, {num_classes}')
    predictor = GenrePredictor(
        model_path='logs/music_cnn_model.pth',
        input_shape=input_shape,
        num_classes=num_classes,     
        genre_map=final_genre_map
    )

    # Путь до любого трека
    test_track = 'Musics/mydataset/Russian Rap/madkid.flac'

    predicted_genres = predictor.predict_genres(test_track)
    
    if predicted_genres:
        print(f"\nПредсказанные жанры для файла '{test_track}':")
        for genre, probability in predicted_genres:
            print(f"  - {genre}: {probability:.2%}")
    else:
        print(f"\nНе удалось предсказать жанры '{test_track}'.")
"""



if __name__ == '__main__':
    genres_df = pd.read_csv('final_dataset/track_genres.csv', index_col=0)
    all_genres_info = pd.read_csv('Musics/fma_metadata/fma_metadata/genres.csv', index_col=0)
    genre_id_to_name = all_genres_info['title'].to_dict()
    genre_columns = genres_df.columns.astype(int).tolist()
    final_genre_map = {i: genre_id_to_name[genre_id] for i, genre_id in enumerate(genre_columns)}
    
    SECONDS_PER_SAMPLE = 5
    SAMPLE_RATE = 22050
    mel_generator = Mel()
    dummy_audio = np.zeros(SAMPLE_RATE * SECONDS_PER_SAMPLE)
    dummy_spectrogram = mel_generator.generate(dummy_audio, SAMPLE_RATE)
    dummy_tensor = torch.tensor(dummy_spectrogram).unsqueeze(0)
    
    input_shape = dummy_tensor.shape
    num_classes = len(genres_df.columns)

    predictor = GenrePredictor(
        model_path='logs/music_cnn_model.pth',
        input_shape=input_shape,
        num_classes=len(genres_df.columns),
        genre_map=final_genre_map
    )
    
    # Путь к аудио
    audio_dir = 'Musics/fma_medium'
    original_tracks_df = pd.read_csv('Musics/fma_metadata/fma_metadata/tracks.csv', header=[0, 1], index_col=0)
    
    val_track_ids_full = original_tracks_df[original_tracks_df[('set', 'split')] == 'validation'].index
    available_val_ids = list(set(val_track_ids_full) & set(genres_df.index))
    
    print(f"{len(available_val_ids)} треков в валидационной выборке")
    NUM_TRACKS_TO_TEST = 5 * 5
    tracks_to_test = random.sample(available_val_ids, NUM_TRACKS_TO_TEST)
    print(f"  ТЕСТ СЛУЧАЙНЫХ ТРЕКОВ")


    for track_id in tracks_to_test:
        import os
        filename = f"{track_id:06d}.mp3"
        sub_folder = f"{track_id:06d}"[:3]
        track_path = os.path.join(audio_dir, sub_folder, filename)

        if not os.path.exists(track_path):
            continue
        predicted_genres = predictor.predict_genres(track_path)

        true_labels_series = genres_df.loc[track_id]
        
        true_genre_indices = true_labels_series[true_labels_series == 1].index
        true_genres = [genre_id_to_name.get(int(genre_id), f"Неизвестный ID {genre_id}") for genre_id in true_genre_indices]

        print(f"Трек: {track_id}")
        print(" Предсказание модели:")
        if predicted_genres:
            for genre, prob in predicted_genres:
                print(f"     - {genre}: {prob:.2%}")
        else:
            print("нет предсказаний выше порога")
            
        print("Реальные жанры:")
        for genre in sorted(true_genres):
            print(f"     - {genre}")
        print("-" * 60)