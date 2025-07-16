import torch
import torch.nn as nn
from tqdm import tqdm


class MusicCNN(nn.Module):
    """
    Сверточная нейронная сеть для многолейбловой классификации жанров музыки
    """
    def __init__(self, input_shape, num_classes):
        super(MusicCNN, self).__init__()

        # Поиск примитивных признаков
        self.conv_block1 = nn.Sequential(
            # Ищем локальные паттерны
            # in_channels=1 - 1 канал
            # out_channels=32 - 32 разных примитивных признаков
            # kernel_size=3 - Ядро 3x3
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1),
            # Нормализация
            nn.BatchNorm2d(32),
            # Функция активации
            nn.ReLU(),

            #пулинг, уменьшает размер "картинки" в 2 раза.
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4) 
        )

        with torch.no_grad():
            input = torch.randn(1, *input_shape)
            output = self._get_conv_output(input)
            # Выпрямляем и получаем размер
            output_size = output.flatten(start_dim=1).shape[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def _get_conv_output(self, x):
        """функция для прогона через сверточные блоки """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Определяет прямой проход данных через модель """
        x = self._get_conv_output(x)
        
        x = x.view(x.size(0), -1)
        #Классифицируем
        logits = self.classifier(x)
        
        return logits

def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device):
    """
    функция для обучения и валидации модели

    Args:
        model (nn.Module): Модель для обучения.
        train_loader (DataLoader): Загрузчик данных для обучения
        val_loader (DataLoader): Загрузчик данных для валидации
        optimizer: Оптимизатор 
        loss_fn: Функция потерь
        num_epochs (int): Количество эпох обучения
        device: Устройство для вычислений ('cuda' или 'cpu')

    Returns:
        dict: Словарь с историей обучения (потери и метрики).
    """
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train() 
        
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"[Training] - epoch {epoch + 1}/{num_epochs}")
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # делаем forward 
            outputs = model(inputs)
            
            # Сравниваем предсказания модели с правильными ответами
            loss = loss_fn(outputs, labels)
            
            # Вычисляем градиенты
            loss.backward()
            
            # Оптимизатор обновляет веса модели, используя вычисленные градиенты
            optimizer.step()
            
            # Собираем статистику по потерям
            running_loss += loss.item() * inputs.size(0)
            
            # Обновляем информацию в прогресс-баре
            train_pbar.set_postfix({'loss': loss.item()})
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Отключаем вычисление градиентов для валидации
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"[Validation] epoch {epoch + 1}/{num_epochs}")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # sigmoid, чтобы получить вероятности
                preds = torch.sigmoid(outputs)
                # порог 0.5: все что выше - 1, ниже - 0
                preds = (preds > 0.5).float()
            
                # Сравниваем предсказания с реальными метками
                total_predictions += labels.numel()
                correct_predictions += (preds == labels).sum().item()

                val_pbar.set_postfix({'loss': loss.item()})

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_predictions / total_predictions
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} -> "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Accuracy: {epoch_val_accuracy:.4f}")

    print("Обучение завершено")
    return history

