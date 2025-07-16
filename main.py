import pandas as pd
from dataset import FmaDataset, split_dataset
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from CNN import MusicCNN, train_model
from visualization import plot_history

"""# Тестируем разные данные входные
height_resolutions = [1024,2048,4096]
hop_lengths = [128,256,512,1024]
n_mels = [256, 512, 1024]
for fft in height_resolutions:
    for hop_length in hop_lengths:
        for mels in n_mels:
            mel_spectrogram = librosa.feature.melspectrogram(y=y, 
                                                             sr=sr,
                                                             n_fft=fft, 
                                                             hop_length=hop_length,
                                                             n_mels = mels)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            plt.figure(figsize=(20, 8))
            librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Мел-спектрограмма')
            plt.tight_layout()
            filename = f"fft_{fft}_hop_{hop_length}_mels_{mels}.png"
            plt.savefig(f'plots/{filename}')
            # plt.show()
"""

#в итоге не использовал
def preprocess(dataset):
    from tqdm import tqdm
    import numpy as np
    import h5py
    import librosa
    from dataset import Mel
    import os
    print("Инициализация датасета для препроцессинга...")

    # Создаем папку для сохранения спектрограмм
    output_dir = "final_dataset/spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Начинаем генерацию и сохранение {len(dataset)} спектрограмм...")
    for i in tqdm(range(len(dataset))):
        try:
            spectrogram, _ = dataset[i]
            # Получаем track_id и sample_num для уникального имени файла
            sample_info = dataset.samples_map[i]
            track_id = sample_info['track_id']
            sample_num = sample_info['sample_num']
            
            filename = f"t_{track_id}_s_{sample_num}.npy"
            filepath = os.path.join(output_dir, filename)
            
            # Сохраняем тензор как NumPy массив
            np.save(filepath, spectrogram.numpy())
        except Exception as e:
            print(f"Ошибка на сэмпле {i}: {e}")
            continue

    print("Препроцессинг завершен!")
    
    '''samples_map = dataset.samples_map
    num_samples = len(samples_map)
    print(f"Всего сэмплов для обработки: {num_samples}")

    HDF5_PATH = 'final_dataset/spectrograms.h5'

    sample_spectrogram, _ = dataset[0]
    spec_shape = sample_spectrogram.shape

    CHUNK_SIZE = 9000
    # Создаем буфер в памяти
    chunk_buffer = np.zeros((CHUNK_SIZE, *spec_shape), dtype='float32')
    chunk_item_count = 0
    start_index = 0


    with h5py.File(HDF5_PATH, 'w') as h5f:
        print(f"Создание HDF5 файла: {HDF5_PATH}")
        
        # (количество_сэмплов, каналы, высота, ширина)
        spectrograms_dset = h5f.create_dataset(
            'spectrograms',
            shape=(num_samples, *spec_shape),
            dtype='float32',
            compression='gzip'
        )
        
        mel_generator = Mel()
        
        for i in tqdm(range(3), desc="Генерация и сохранение спектрограмм"):
            try:
                sample_info = samples_map[i]
                track_id = sample_info['track_id']
                sample_num = sample_info['sample_num']
                
                file_path = dataset._get_file_path(track_id)
                offset = sample_num * dataset.seconds_split
                
                y, sr = librosa.load(file_path, offset=offset, duration=dataset.seconds_split)
                
                spectrogram = mel_generator.generate(y, sr)
                spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float16).unsqueeze(0)
                print(spectrogram_tensor)
                # chunk_buffer[chunk_item_count] = spectrogram_tensor.numpy()
                # chunk_item_count += 1

                # # Если буфер заполнился, записываем его
                # if chunk_item_count == CHUNK_SIZE:
                #     end_index = start_index + CHUNK_SIZE
                #     spectrograms_dset[start_index:end_index] = chunk_buffer
                    
                #     # Сбрасываем счетчики
                #     start_index = end_index
                #     chunk_item_count = 0
                
            except Exception as e:
                print(f"Пропущен сэмпл {i} (track {track_id}) из-за ошибки: {e}")
                spectrograms_dset[i] = np.zeros(spec_shape, dtype='float32')
        
        # Если осталась часть в буфере
        if chunk_item_count > 0:
            end_index = start_index + chunk_item_count
            # Записываем только заполненную часть буфера
            spectrograms_dset[start_index:end_index] = chunk_buffer[:chunk_item_count]

    print("Препроцессинг завершен")'''


if __name__ == '__main__':
    # обьявляем логи
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers=[
            logging.FileHandler('logs/data_processing.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # путь к корневой папке песен
    audio_dir_path = 'Musics/fma_medium' 
    tracks_df = pd.read_csv('final_dataset/track_genres.csv',index_col=0)

    print(f'Загружено - {len(tracks_df)} треков')
    logger.info(f"Загружено - {len(tracks_df)} треков")
    
    MusicsDataset = FmaDataset(audio_dir_path, tracks_df, seconds_split = 5)
    # preprocess(MusicsDataset)

    # создание train_loader и val_loader 
    train_loader, val_loader = split_dataset(MusicsDataset) 


    sample_spectrogram, _ = MusicsDataset[0] 
    input_shape = sample_spectrogram.shape
    num_classes = len(tracks_df.columns)
    logger.info(f"Параметры для модели: input_shape={input_shape}, num_classes={num_classes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")
    model = MusicCNN(input_shape=input_shape, num_classes=num_classes).to(device)

    # Функция потерь для многолейбловой классификации
    loss_fn = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 1
    logger.info(f"Начало обучения на {NUM_EPOCHS} эпох")
    history = train_model(model, train_loader, val_loader, optimizer, loss_fn, NUM_EPOCHS, device)

    # Сохраняем обученные веса модели
    torch.save(model.state_dict(), 'logs/music_cnn_model.pth')
    logger.info("Модель сохранена в файл 'music_cnn_model.pth'")
    
    # Рисуем графики
    plot_history(history)


