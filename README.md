# Analysis-of-weld-defects-using-neural-networks
Analysis of weld defects using neural networks. Программа была разработана для облегчения обнаружения и классификации дефектов сварных швов. Программа работает с рентгенографическими изображениями, полученные одним из методов неразрушающего контроля, а именно метод радиационного контроля и полностью написана на Python. Изображения были взяты из открытого датасета GDXray.
Сначала происходит обнаружение дефектов с помощью множественных функций переобразований, которые были взяты в библиотеке OpenCV. Основными функциями являются адаптивная бинаризация, морфологическая операция - открытие и поиск контуров. Изображения различны и для использования некоторых функций изображения классифицируются методом kNN.
После того как дефекты были обнаружены, они вырезаются из начального изображения и передаются готовой модели CNN для классификации. Модель способна классифицировать 5 классов: поры, шлаковые включения, без дефектов, продольные и поперечные трещины.
За основу модели CNN была взята VGG-16 и исполовалась технология перенос обучения. Датасет состоит из 1270 участков изображений с определенным дефектом. Обучение проходило 16 эпох с точность примерно 81%. Использовались библиотеки Trnsorflor и Keras на платформе Google Colaboratory.
