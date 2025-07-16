import matplotlib.pyplot as plt

def plot_history(history, title='История обучения'):
    """
    Рисует графики потерь и точности для обучающей и валидационной выборок

    Args:
        history (dict): 'train_loss', 'val_loss', 'val_accuracy'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', color='yellow')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.set_title('График потерь')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history['val_accuracy'], label='Val Accuracy', color='green')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность (Accuracy)')
    ax2.set_title('График точности на валидации')
    ax2.legend()
    ax2.grid(True)

    # Сохраняем график в файл
    plt.savefig('logs/training_history.png')
    print("\nГрафик истории обучения сохранен в файл 'logs/training_history.png'")
    
    # Показываем график
    plt.show()