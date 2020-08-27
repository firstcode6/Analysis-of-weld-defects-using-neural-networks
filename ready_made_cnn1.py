import numpy as np
from tensorflow.keras.models import model_from_json
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
#import matplotlib.pyplot as plt
import tensorflow as tf
#import cv2


class CNN_Classification:

    # loaded_model= model_from_json
    # конструктор
    def __init__(self):
        #print("lol")

        # Список классов
        self.classes = ['Без дефекта', 'Трещина продольная(гориз)', 'Пора', 'Шлаковые включения',
                        'Трещина поперечная(верт)']

        """Загружаем обученную модель
        weld_vgg19_1.h5 - сохраняет 
        архитектура модели, позволяющая воссоздать модель
        вес модели
        конфигурация тренировки (потеря, оптимизатор)
        состояние оптимизатора, позволяющее возобновить обучение именно с того места, где вы остановились.

        weld_vgg19_1.json-  сохранить архитектуру модели , а не ее вес или конфигурацию обучения, вы можете сделать:
        """
        json_file = open("weld_vgg16_5_classes254_20_128.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)  # загружаем веса
        self.loaded_model.load_weights("weld_vgg16_5_classes254_20_128.h5")

        # from tensorflow.keras.models import load_model
        # loaded_model = load_model("C:/Users/1/Desktop/pictures/save_model/weld_vgg19_1.h5")

        """Компилируем модель"""

        self.loaded_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                                  loss='categorical_crossentropy',
                                  metrics=["accuracy"])

    def classification_of_defects(self, array_images):
        """Загружаем изображение для распознавания"""
        predict_cl = []
        predict_accuracy = []
        for i in range(len(array_images)):
            '''Сеть ожидает одно или несколько изображений в качестве входных данных; это означает, что входной массив должен быть четырехмерным: сэмплы, строки, столбцы и каналы.'''

            x = np.expand_dims(array_images[i], axis=0)  # добавляет измерение(3,3)->(1,3,3)
            # print(x)
            '''Затем пиксели изображения должны быть подготовлены так же, как были подготовлены обучающие данные ImageNet.
             В частности, из бумаги:
             Единственная предварительная обработка, которую мы делаем, - это вычитание среднего значения RGB, 
             вычисленного на обучающем наборе, из каждого пикселя.'''
            x = preprocess_input(x)

            """Запускаем распознавание"""

            prediction = self.loaded_model.predict(x)

            #predict_cl.append(max(prediction))

            #print((i + 1), " prediction= ", prediction)

            # print(self.classes[np.argmax(prediction)])

            # Возвращает индексы максимальных значений по оси.
            predict_cl.append(np.argmax(prediction))

            # Возвращает максимальных значений по оси.  a = float('{:.3f}'.format(x))

            accurracy=np.max(prediction)
            #accurracy
            predict_accuracy.append(accurracy )
            # print("check = ",self.classes[1]) # Без дефекта
        return predict_cl, predict_accuracy


# p = CNN_Classification
