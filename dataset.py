from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import torch
import os
import pandas as pd 
import ast
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pickle
import logging
from torch.utils.data import random_split, DataLoader

# df = pd.read_csv('fma_metadata/fma_metadata/tracks.csv', header=[0, 1], index_col=0)
# genre_counts = df[('track', 'genres_all')].value_counts()
# # print(df.head())
# train_df = df[df[('set', 'split')] == 'training']
# test_df = df[df[('set', 'split')] == 'test']
# print(f'{train_df.info()}')
# print(f'{test_df.info()}')
logger = logging.getLogger(__name__)

def clear_dataset(old_path, new_path):
    try:
        # header=[0, 1] первые две строки - это заголовки
        # index_col=0 первая колонка - это индекс
        tracks_df = pd.read_csv(old_path, header=[0, 1], index_col=0)

        genre_series = tracks_df[('track', 'genres_all')]
        # track_genre_map = genre_series.reset_index()
        # track_genre_map.columns = ['track_id', 'genres_all']
        # print(f'----------------------------')
        # print(len(track_genre_map))
        genre_series = genre_series.dropna()
        # print(len(track_genre_map))
        # print(f'----------------------------')
        # преобразуем строки '[17, 10, ...]' в реальные списки чисел 
        # genre_series['genres_list'] = genre_series['genres_all'].apply(ast.literal_eval)
        try:
            genre_series = genre_series.apply(ast.literal_eval)
        except ValueError:
            print("некоторые ячейки в столбце 'genres_all' имеют неверный формат")

        mlb = MultiLabelBinarizer()
        one_hot_genres = pd.DataFrame(
                                    mlb.fit_transform(genre_series),
                                    columns=mlb.classes_,  
                                    index=genre_series.index
                                    )
        # print("\n Финальный")
        # print(one_hot_genres.head())
        # print(one_hot_genres.columns)
        # print(f"Название первой колонки: {one_hot_genres.columns[0]}")
        print(one_hot_genres.info())
        one_hot_genres.to_csv(new_path)
        print(f"\n сохранен в файл {new_path}")

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути ")
    except KeyError:
        print("Ошибка: Не удалось найти столбцы. Убедитесь, что структура файла верна.")

class Mel():
    def __init__(self,height_resolution = 2048,hop_length = 128, n_mels = 256):
        self.height_resolution = height_resolution
        self.hop_length = hop_length
        self.n_mels = n_mels

    def generate(self, y, sr):
        """
        Генерирует спектрограмму из аудио и возвращает
        Создание спектограмы 2D из 1D numpy массива
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr,
                                                        n_fft=self.height_resolution,
                                                        hop_length=self.hop_length,
                                                        n_mels=self.n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return log_mel_spectrogram

    def show_mel(self, spectogram, sr, is_save = False, is_Bar = False):
        """
        Построение и показ мел-спектограммы для аудио
        """
        plt.figure(figsize=(20, 8))
        librosa.display.specshow(spectogram, sr=sr, x_axis='time', y_axis='mel', hop_length = self.hop_length)
        if is_Bar:
            plt.colorbar(format='%+2.0f dB')
        plt.title('Мел-спектрограмма')
        plt.tight_layout()
        if is_save:
            filename = f"mel_spectogram.png"
            plt.savefig(f'plots/{filename}')
        plt.show()

'''
def music_splitter(music, Hz, seconds_part = 10):
    """
    Разделяет музыку на равные куски по ... секунд и вовзращает список ndarray

    Args:
        music (ndarray): Список звуковой волны/ сэмплов
        Hz (int, float): Частота дискретизации
        seconds_part (int, optional): Значение отрезков, на которые будет делится музыка
    """
    parts = []
    music_len = len(music) / Hz
    chunks = int(music_len // seconds_part)
    for part in range(chunks):
        start = part * seconds_part * Hz # Получаем начальный индекс 
        end = (part + 1) * seconds_part * Hz # Конечный индекс
        parts.append(music[start:end])

    return parts
'''

class FmaDataset(Dataset):
    """
    Инициализирует датасет
    """
    def __init__(self, audio_dir, genres_df, seconds_split = 5, samples_path="final_dataset/samples_map.pkl"):
        """
        Args:
            audio_dir (str): путь к корневой папке с аудио 
            genres_df (pd.DataFrame): one-hot encoded DataFrame, индекс - track_id
        """
        self.audio_dir = audio_dir
        self.genres_df = genres_df
        self.seconds_split = seconds_split
        self.samples_path = samples_path
        
        # track_id
        self.track_ids = self.genres_df.index.tolist()
        self.labels = self.genres_df.values

        self.samples_map = []
        if os.path.exists(self.samples_path):
                logger.info(f"Загрузка сэмплов из {self.samples_path}")
                with open(self.samples_path, 'rb') as f:
                    self.samples_map = pickle.load(f)
        else:
            logger.info("Сохранение не найдено, создание нового файла сэмплов")
            self.samples_map = self._create_samples_map()
            logger.info(f"Сохранение сэмплов в: {self.samples_path}")
            with open(self.samples_path, 'wb') as f:
                pickle.dump(self.samples_map, f)     
        logger.info(f"Всего сэмплов: {len(self.samples_map)}")

    def _create_samples_map(self):
        temp_map = []
        error_count = 0
        from tqdm import tqdm 
        for track_id in tqdm(self.track_ids, desc="Создание сэпмлов"):
            try:
                file_path = self._get_file_path(track_id)
                duration = librosa.get_duration(path=file_path)
                num_samples = int(duration // self.seconds_split)
                
                for i in range(num_samples):
                    temp_map.append({'track_id': track_id, 'sample_num': i})
            except Exception as e:
                error_count += 1
                # logger.info(f"В пути: {file_path} не обработался трек {track_id} {i} сэмплом c ошибкой: {e}")
                pass
        if error_count > 0:
            logger.warning(f"Завершено с {error_count} ошибками ")
        return temp_map

    def __len__(self):
        return len(self.samples_map)

    def _get_file_path(self, track_id):
        """
        Формирует путь до трека по айди трека
        """
        filename = f"{track_id:06d}.mp3"
        sub_folder = f"{track_id:06d}"[:3]
        return os.path.join(self.audio_dir, sub_folder, filename)

    def __getitem__(self, idx):
        '''
        При вызове элемента(трека) возвращает его спектограму и верные жанры
        '''
        sample_info = self.samples_map[idx]
        track_id = sample_info['track_id']
        sample_num = sample_info['sample_num']

        # загружаем трек и считаем смещение(по номеру сэмпла)
        file_path = self._get_file_path(track_id)
        offset = sample_num * self.seconds_split
        try:
            y, sr = librosa.load(file_path, 
                                 offset=offset, 
                                 duration=self.seconds_split)
            
            # Создаем спектрограмму из этого куска
            mel = Mel() 
            spectrogram = mel.generate(y, sr)
            spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
            #ищем по айди трека и забираем значения
            labels = self.genres_df.loc[track_id].values
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            
            return spectrogram_tensor, labels_tensor

        except Exception as e:
            # print(f"Ошибка при обработке сэмпла {idx} (track {track_id}): {e}")
            return self.__getitem__((idx + 1) % len(self))


def split_dataset(dataset, val_split_ratio=0.2, batch_size=32, num_workers=1):
    """
    Разделяет датасет на обучающую и валидационную выборки

    Args:
        dataset (torch.utils.data.Dataset): Полный датасет
        val_split_ratio (float): процент данных для валидационных данных(от 0 до 1)
        batch_size (int): Размер батчей
        num_workers (int): Количество параллельных процессов для загрузки

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Чтобы при каждом запуске разделение было одинаковым
    torch.manual_seed(42)

    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split_ratio)
    train_size = dataset_size - val_size
    print(f"Разделение датасета ({dataset_size} сэмплов):")
    print(f"    Обучающая выборка: {train_size} сэмплов")
    print(f"    Валидационная выборка: {val_size} сэмплов")
    train_sub, val_sub = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True # Ускоряет передачу данных на GPU
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader



# clear_dataset('fma_metadata/fma_metadata/tracks.csv', 'final_dataset/track_genres.csv')