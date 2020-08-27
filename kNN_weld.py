import numpy as np
import mahotas as mh
from glob import glob
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.decomposition import PCA


class preparing_data_for_kNN:
    # def __init__(self):

    '''
    a = [10, 20] enumerate -> (0, 10) (1, 20)
    glob(path) считывает все папки '/content/drive/My Drive/Colab Notebooks/классы1/7_4х4', 9_3х3', 7_3х3', 5_2х2' ...
    sorted(glob(path)) сортирует из по возрастанию 5_2х2 5_3x3'...
    dir_=/content/drive/My Drive/Colab Notebooks/классы1/5_2х2 и остальные
    i-итерация=кол-во папок
    '''

    # считываем изображения

    # нормализовать от 0 до 1
    def normalize(self, img):
        im = img - np.min(img)
        return im / np.max(im)

    '''Вычислить особенности текстуры Харалика
    Вычисляет особенности текстуры Харалика для четырех двухмерных направлений или тринадцати трехмерных направлений (в зависимости от размеров f ).'''

    def extract_haralick_features(self, image):
        # расчёт текстурных признаков Харалика
        # возвращает текстуры 13 массив в каждом 13 чисел
        textures = mh.features.haralick(image)
        # print(textures)
        # сокращаем до 1 массива с 13 средними числами
        ht_mean = textures.mean(axis=0)
        # print(ht_mean)
        # возвращает массив из 13 усредненых значений текстуры для одного изображения
        return ht_mean

    # получение признаков для изображений
    def extract_features_for_imgs(self, images, several_images):
        # несколько изображений
        if several_images == True:
            # извлекаем векторы признаков(текстуры) для каждого изображения
            # получается массив=кол-во изоб в каждом 13 значений текстур
            feat = np.array([self.extract_haralick_features(img) for img in images])

            # добавляю ещё дисперсию в качестве фичи
            # вычисляем дисперсию для каждого отдельного изображения, получаем массив из числе=кол-ву изображений
            std_feat = np.array([np.std(self.normalize(img)) for img in images]).reshape(-1, 1)
            # print("std_feat2=", std_feat)
        # одно изображение
        elif several_images == False:
            feat = np.array(self.extract_haralick_features(images))
            # обернем его массивом
            feat = np.array([feat])
            std_feat = np.array(np.std(self.normalize(images))).reshape(-1, 1)

        # сливаем два массива в одном элементе будет 13+1=14 значений текстур для каждого изображения
        feat = np.hstack((feat, std_feat))
        return feat

    def textual_features_of_Haralik(self, bool_traning_data, processed_img):

        features = [] # тестурные данные для обучений состоит из 34изоб * 14 признаков
        classes_label = []  # это метки классов

        # если данные нужно считать заново то
        if bool_traning_data == True:
            classes_img = []  # это картинки из "классов"

            path = 'C:/Users/1/Desktop/pictures/classes/*'

            for i, dir_ in enumerate(sorted(glob(path))):
                # dir_=папке одного класса, дальше считываем файлы с папки и сохраняем их в лист
                img_list = [plt.imread(img_path) for img_path in glob(dir_ + '/weld*')]

                # добавляем в лист классы изображений, индекс(i)=номер класса * на кол-во изоб в папке действующей итериции
                classes_label.extend([i] * len(img_list))

                # записали изображения в лист все(2,3,9,11,3,6)=34
                classes_img.extend(img_list)

            # # название классов сортированно ['5_2х2' '5_3x3' '7_2х2' '7_3х3' '7_4х4' '9_3х3']
            # classes_names = np.array([p.split('/')[-1] for p in sorted(glob(path))])
            # print(classes_names)

            # получаем массив=кол-ву изображений, в одном массиве хварится 14 чисел текстр именного этого изображения
            features = self.extract_features_for_imgs(classes_img, True)

            # сохраняем вычисления
            np.save('features', features)
            np.save('classes_label', classes_label)
        # данные для обучения сохраненные
        elif bool_traning_data == False:

            features = np.load('features.npy')
            classes_label = np.load('classes_label.npy')

        # вычисляем текстры для 1 нового изображения  False-одно изображение
        features_img = self.extract_features_for_imgs(processed_img, False)

        return features, classes_label, features_img


# _______________________________________________Knn__________________________________________
class kNN:

    # def __init__(self):
    #     print("lol")

    # теорема Пифагора на максималках
    def evklid_metric(self, a, b):
        try:
            len(a)
        except TypeError:
            a, b = np.array([a]), np.array([b])

        # получаем два массива с 14 элементами, каждый с каждым отнимаем по интексу и суммируем разницу и вывод кв. корень
        # получаем евклидовое расстояние для 14 элементов
        result = np.sum([(a[i] - b[i]) ** 2 for i in range(0, len(b))]) ** 0.5
        #print("result = ", result)
        return result

    # (данные для обучения, данные для проверки, k- кол-во ближайших точек, кол-во классов)

    def funct_kNN(self, x_train, y_train, x_test, k):
        y_test = []
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        for a in x_test:
            # вычисляем евклидовое расстояние для каждой текстуры
            dists_to_x_train = [self.evklid_metric(a, b) for b in x_train]
            # print("dists_to_x_train = ",dists_to_x_train)

            # argsort()Возвращает индексы, которые будут сортировать массив. np.array([7, 3, 5]).argsort() -> array([1, 2, 0])
            # # [0:k]-берем объект от 0 до k, k=количество точек которые мы возьмем для подсчета k=2-> [2,0],[5,1]
            top_ind = np.array(dists_to_x_train).argsort()[:k]
           # print("top_ind = ", top_ind)


            top_classes = list(y_train[top_ind])
            #print("top_classes = ", top_classes)
            target = max(top_classes, key=top_classes.count)
            y_test.append(int(target))
        return np.array(y_test)

    def defining_class(self, img):
        # объект класса для получения данных от изображениях
        object_preparing = preparing_data_for_kNN()

        # # данные для теста
        # test_patch = 'C:/Users/1/Desktop/pictures/класс/*'
        # test_imgs = [plt.imread(img1) for img1 in sorted(glob(test_patch))]
        # print("type(test_imgs[0]) = ", type(test_imgs[0]))
        # trainData, classes_label, imgData = object_preparing.textual_features_of_Haralik(False, test_imgs[1])
        # k = 1  # k ближайших
        # predict_np = self.funct_kNN(trainData, classes_label, imgData, k)
        # print(predict_np)

        '''
        # отправляем метку что обучающий набор нужен и тестовае изображения
         # trainData -данные для обучения массив из 34изображений х 14текстр а их классы = classes_label
           imgData = 14чисел(тестуры) исследуемого массива
         # (используем сохраненные данные для обучения=False, рассчитаем заново=True, 1изоб которые получаем параметры)'''
        trainData, classes_label, imgData = object_preparing.textual_features_of_Haralik(False, img)

        # print("trainData=", trainData)
        # print("imgData=", imgData)
        # print("classes_label=", classes_label)


        # предсказание на изображениях, на которых обучалось и KNN numpy
        k = 1  # k ближайших
        predict_np = self.funct_kNN(trainData, classes_label, imgData, k)

        # перевод в int массива с одним значениемберем из массива 1 число
        predict_np = predict_np[0]

        print("класс = ", predict_np)
        return predict_np


# import cv2
#
# img_path = "C:/Users/1/Desktop/pictures/weld28.JPG"
# img = cv2.imread(img_path)  # <class 'numpy.ndarray'>
#
# object_kNN = kNN()
#
# object_kNN.defining_class(img)
