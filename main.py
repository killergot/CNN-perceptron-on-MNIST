import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Предобработка данных
# Приводим значения пикселей к диапазону от 0 до 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Изменяем форму массива для работы с Keras: добавляем канал (1 для градаций серого)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Преобразуем метки классов в категориальный формат (One-Hot Encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Вывод информации о подготовленных данных
print("Форма x_train:", x_train.shape)
print("Форма y_train:", y_train.shape)
print("Форма x_test:", x_test.shape)
print("Форма y_test:", y_test.shape)

model_path = (r'B:\Programm\python\lab_bai\lab2\my_model.keras')

def load_trained_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    print(f'Модель загружена из: {model_path}')
    return loaded_model

# def show_mnist_image(index):
#     """
#     Функция для отображения изображения из набора данных MNIST по его номеру.
#
#     Параметры:
#         index (int): Номер изображения от 1 до 60000, где:
#                      - первые 60000 изображений - тренировочная выборка.
#
#     """
#     # Проверка, что номер находится в допустимом диапазоне
#     if 1 <= index <= 60000:
#         # Определение, обучающая или тестовая выборка
#         if index <= len(x_train):
#             image = x_train[index - 1]
#             label = y_train[index - 1]
#         else:
#             image = x_test[index - len(x_train) - 1]
#             label = y_test[index - len(x_train) - 1]
#
#         # Отображение изображения и его метки
#         plt.imshow(image, cmap='gray')
#         plt.title(f'Метка: {label}')
#         plt.axis('off')
#         plt.show()
#     else:
#         print("Ошибка: номер должен быть в диапазоне от 1 до 60000")
#
#
# # Пример вызова функции
# show_mnist_image(60000)
# Построение модели
model = Sequential()

# Первый сверточный слой
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Второй сверточный слой
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Преобразование данных в плоский вид и полносвязные слои
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Регуляризация
model.add(Dense(10, activation='softmax'))  # Выходной слой для классификации на 10 классов

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)
model = load_trained_model(model_path)
# Оценка модели
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Точность на тестовых данных: {test_accuracy * 100:.2f}%')

# Получение предсказаний для тестового набора
y_pred_proba = model.predict(x_test)  # Предсказания вероятностей для каждой метки
y_pred = np.argmax(y_pred_proba, axis=1)  # Классовые предсказания
y_test_labels = np.argmax(y_test, axis=1)  # Преобразование y_test в одномерный массив меток

# Вычисление метрик
accuracy = accuracy_score(y_test_labels, y_pred)
precision = precision_score(y_test_labels, y_pred, average='macro')
recall = recall_score(y_test_labels, y_pred, average='macro')
f1 = f1_score(y_test_labels, y_pred, average='macro')

# Специфичность для каждого класса
cm = confusion_matrix(y_test_labels, y_pred)
specificity = []
for i in range(len(cm)):
    tn = np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]
    fp = np.sum(cm[:, i]) - cm[i, i]
    specificity.append(tn / (tn + fp))
average_specificity = np.mean(specificity)

# Вычисление ROC-AUC для многоклассовой классификации
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision (macro): {precision:.4f}')
print(f'Recall (macro): {recall:.4f}')
print(f'F1-score (macro): {f1:.4f}')
print(f'Specificity (average): {average_specificity:.4f}')
print(f'ROC AUC (macro): {roc_auc:.4f}')

# Построение ROC-кривой для каждого класса
import matplotlib.pyplot as plt

fpr = {}
tpr = {}
for i in range(10):  # Для каждого класса
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])

# Построение графика ROC-кривой для каждого класса
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label=f'Класс {i}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая для каждого класса')
plt.legend(loc='lower right')
plt.show()
def save_trained_model(model, model_path):
    model.save(model_path)
    print(f'Модель сохранена в: {model_path}')

# save_trained_model(model, model_path)


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1) 
y_true = np.argmax(y_test, axis=1)        

conf_matrix = confusion_matrix(y_true, y_pred_classes)

print("Матрица путаницы:")
print(conf_matrix)

plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица путаницы')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.show()



def predict_image(model, image_path):
    """
    Функция для выполнения предсказания на одной картинке с использованием обученной модели.

    Параметры:
        model: обученная модель TensorFlow/Keras.
        image_path (str): путь к изображению.

    Возвращает:
        предсказанный класс и массив вероятностей для каждого класса.
    """
    # Загружаем изображение и преобразуем его к размеру 28x28
    img = image.load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    # Преобразуем изображение в массив numpy
    img_array = image.img_to_array(img)
    # Нормализация: преобразование значений пикселей к диапазону [0, 1]
    img_array = img_array.astype('float32') / 255.0
    # Изменение формы массива для передачи в модель (добавляем ось batch)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28, 1)
    # Выполняем предсказание
    predictions = model.predict(img_array)
    # Определение класса с максимальной вероятностью
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probabilities = predictions[0]
    print(f'Предсказанный класс: {predicted_class}')
    print(f'Вероятности по классам: {predicted_probabilities}')

    return predicted_class, predicted_probabilities


for i in range(10):
    print(f'Моя циферка {i}\n')
    predict_image(model, f'{i}.png')