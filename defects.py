import numpy as np
import cv2
import matplotlib.pyplot as plt
from ready_made_cnn1 import CNN_Classification

from kNN_weld import kNN
from PIL import Image


class defects_detection:
    def __init__(self):
        self.contours = []

    def viewImage(self, image):
        cv2.imshow('Display', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def search_for_defects(self, img, num_class):

        list_classes = [[5, 2], [5, 4], [7, 2], [7, 3], [7, 4], [9, 3]]

        size_pixel = list_classes[num_class][0]
        size_open = list_classes[num_class][1]

        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # self.viewImage(gray)

        # адаптивное выравнивание гистограммы
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray1 = clahe.apply(gray)
        # viewImage(gray1)

        gauss = cv2.GaussianBlur(gray1, (3, 3), 0)  # сглаживая матрицой 5х5
        gauss1 = cv2.GaussianBlur(gauss, (5, 5), 0)  # сглаживая матрицой 5х5

        threshold = cv2.adaptiveThreshold(gauss1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, size_pixel, 5)
        # viewImage(th2)

        kernel = np.ones((size_open, size_open), np.uint8)
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        # viewImage(opening)

        pyDown = cv2.pyrDown(opening)
        # viewImage(pyDown)
        pyUp = cv2.pyrUp(pyDown)
        # viewImage(pyUp)

        threshold1 = cv2.adaptiveThreshold(pyUp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 27, 5)
        # viewImage(th2)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(threshold1, cv2.MORPH_OPEN, kernel)
        # viewImage(opening)

        self.contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # лист массивов изображений
        array_images = []
        # цикл по контурам
        for c in self.contours:
            x, y, w, h = cv2.boundingRect(c)  # Draw a straight rectangle with the points

            # Region Of Interest
            ROI = img_copy[y:y + h, x:x + w]
            array_images.append(ROI)

            # cv2.rectangle (изображение, начальная точка, конечная точка, цвет, толщина)
            img_box = cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

            # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            # # (изоб, текст, координаты центра(x,y),тип шрифта,коэффициент масштабирования шрифта,цвет,толщина)
            # cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        return array_images

        # функция возвращаем лист с обработанными изображениями в виде массива numpy  для нейронной сети
        # а получает лист с начальными дефектами

    def preparing_defects(self, array_images):
        # лист для обработанных изображений
        ready_array = []
        for i in range(len(array_images)):
            # перевод np.array в изображение
            img = Image.fromarray(array_images[i], 'RGB')

            # plt.imshow(img)
            # plt.show()

            # длинна и ширина изображения
            height, width = img.size

            # коффициент по которому сравниваем сразаем длинну изображение
            scale_num = 3
            # если длинна больше ширины трех раз
            if height / width > scale_num:
                # (левая верхная точка, правая нижняя точка)
                area = (0, 0, width * scale_num, width)
                img = img.crop(area)
            # если ширина больше длинна трех раз
            elif width / height > scale_num:
                area = (0, 0, height, scale_num * height)
                img = img.crop(area)

            # размеры образанного изображения
            height1, width1 = img.size

            if (max(height1, width1) > 128):
                # коэффицент для соотношение сторон, где максимальна длинна которой = 128
                scale = 128 / max(height1, width1)
                #  возвращает копию этого изображения с измененным размером и улучшенным качеством
                img = img.resize((int(height1 * scale), int(width1 * scale)), Image.ANTIALIAS)

            # перевод изоб в массив numpy
            arr = np.array(img)

            # создаем массив изображения 128х128 белого цвета
            img2 = np.full((128, 128, 3), 255, dtype=np.uint8)  # create
            # изменяем белые пиксели нашим изображением для нейронной сети
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    img2[i][j] = arr[i][j]
            # добавляем изображение в лист
            ready_array.append(img2)
            #  img_path = "C:/Users/1/Desktop/pictures/weld" + nub +".JPG" .save("C:/Users/1/Desktop/pictures/defect"+str(i)+".jpg")
            # img = Image.fromarray(img2, 'RGB')
            # plt.imshow(img)
            # plt.show()

        # возвращаем лист с изображениями в виде массива numpy
        return ready_array

    def draw_defects(self, img, class_of_defects, predict_accuracy):
        # list_colors=['']

        # class_of_defects = [2, 3, 3, 1, 4]
        # predict_accuracy = [0.99989986, 0.97503054, 1.0, 1.0, 1.0]
        #  ['Без дефекта', 'Трещина продольная(гориз)', 'Пора', 'Шлаковые включения','Трещина поперечная(верт)']

        # Blue=0  Green=1 Red=2  Purple фиолетовый=3 Chocolate=4
        # Для BGR мы передаем кортеж.
        colors = [(0, 255, 255), (255, 0, 0), (128, 0, 128), (0, 128, 0), (0, 0, 128)]
        text_cl = ['no defect:', 'l.crack:', 'porosity:', 'slag:', 't.crack:']

        # закрашиваем в нужные цвета дефекты len(contours)=число дефектов
        # print("len(self.contours)= ", len(self.contours))
        for i in range(len(self.contours)):
            # от контура получаем координаты дефекта
            x, y, w, h = cv2.boundingRect(self.contours[i])  # Draw a straight rectangle with the points

            # cv2.rectangle (изображение, начальная точка, конечная точка, цвет, толщина)
            # рисуем прямоугольники, а цвет = классу
            cv2.rectangle(img, (x, y), (x + w, y + h), color=colors[class_of_defects[i]], thickness=2)

            # округляем до двух чисел после запятой
            accuracy_temp = round(predict_accuracy[i], 2)

            # текст, название дефекта+процент
            text = text_cl[class_of_defects[i]] + str(accuracy_temp)  # round(2.55, 2)

            # (изоб, текст, координаты центра(x,y),тип шрифта,коэффициент масштабирования шрифта,цвет,толщина)
            cv2.putText(img, text, (x, y - 4), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=colors[class_of_defects[i]], thickness=1)

        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        return img


class main:

    def main_function(self, img_path, bool_CNN):

        self.class_of_defects = []

        img = cv2.imread(img_path)  # <class 'numpy.ndarray'>

        object_kNN = kNN()
        num_class = object_kNN.defining_class(img)
        print("num_class=", num_class)

        object_def_det = defects_detection()
        array_images = object_def_det.search_for_defects(img, num_class)

        # если дефектов не обнаружено
        if not array_images:
            # print("Нет дефектов")
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            return False, img, self.class_of_defects
        else:
            # если вызвали классифицировать то делаем это
            if (bool_CNN == True):
                ready_array = object_def_det.preparing_defects(array_images)

                # создаем экземпляр класса нейронной сети
                object_cnn = CNN_Classification()
                # обращаемся методу нейронной сети для классификации выявленных дефектов
                # возвращает лист с номерами класса дефекта для каждого изображения
                self.class_of_defects, predict_accuracy = object_cnn.classification_of_defects(ready_array)
                # print(class_of_defects)
                # print(predict_accuracy)
                img_ready = object_def_det.draw_defects(img, self.class_of_defects, predict_accuracy)
                cv2.imshow("Image", img_ready)
                cv2.waitKey(0)
                return True, img_ready, self.class_of_defects
            # если вызвали проверить наличие дефектов то делаем это
            else:
                return True, img, self.class_of_defects


# ob = main()
# img_path = "C:/Users/1/Desktop/pictures/weld5.JPG"  # C:/Users/1/Desktop/pictures
# ob.main_function(img_path, True)
