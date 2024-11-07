import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import random

# Загрузка и предобработка данных
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Словарь для хранения метрик по экспериментам
metrics_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': [], 'roc_auc': []}

# Тестовый массив, где
# (эпохи, Кол-во фильтров, кол-во нейронов, dropouts, скорость обучения)
test = [(2,32,128,0.5,0.0005),
        (10,32,128,0.5,0.0005),
        (5,16,128,0.5,0.0005),
        (5,64,128,0.5,0.0005),
        (5,32,64,0.5,0.0005),
        (5,32,256,0.5,0.0005),
        (5,32,128,0.3,0.0005),
        (5,32,128,0.7,0.0005),
        (5,32,128,0.5,0.0001),
        (5,32,128,0.5,0.001)]

# Проведение экспериментов
for i in range(10):
    # Случайный выбор гиперпараметров
    epochs,filters,dense_unit,dropout_rate,learning_rate = test[i]

    # Построение модели
    model = Sequential([
        Conv2D(filters, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters * 2, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(dense_unit, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Обучение модели
    model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=0)

    # Предсказания на тестовых данных
    y_pred_proba = model.predict(x_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Вычисление метрик
    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred, average='macro')
    recall = recall_score(y_test_labels, y_pred, average='macro')
    f1 = f1_score(y_test_labels, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    # Вычисление специфичности
    cm = confusion_matrix(y_test_labels, y_pred)
    specificity = []
    for j in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[j]) - np.sum(cm[:, j]) + cm[j, j]
        fp = np.sum(cm[:, j]) - cm[j, j]
        specificity.append(tn / (tn + fp))
    average_specificity = np.mean(specificity)

    # Сохранение результатов метрик для текущего эксперимента
    metrics_results['accuracy'].append(accuracy)
    metrics_results['precision'].append(precision)
    metrics_results['recall'].append(recall)
    metrics_results['f1'].append(f1)
    metrics_results['specificity'].append(average_specificity)
    metrics_results['roc_auc'].append(roc_auc)

    print(
        f'Эксперимент {i + 1}: Filters={filters}, Dense Units={dense_unit}, Dropout={dropout_rate}, LR={learning_rate}')
    print(
        f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Specificity: {average_specificity:.4f}, ROC AUC: {roc_auc:.4f}')
    print('---')

# Построение графиков зависимости метрик от экспериментов
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
metrics = list(metrics_results.keys())
for i, ax in enumerate(axs.flat):
    ax.plot(range(1, 11), metrics_results[metrics[i]], marker='o', linestyle='-')
    ax.set_title(f'{metrics[i].capitalize()} по экспериментам')
    ax.set_xlabel('Номер эксперимента')
    ax.set_ylabel(metrics[i].capitalize())
    ax.grid()

plt.tight_layout()
plt.show()
