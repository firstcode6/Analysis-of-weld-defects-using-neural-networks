import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog as fd
from tkinter import ttk
from datetime import date
import sqlite3
from tkinter import messagebox
import pkg_resources.py2_warn



import numpy as np
import mahotas as mh
from glob import glob
import matplotlib.pyplot as plt

# import tensorflow as tf
# import keras
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.applications.vgg19 import preprocess_input


import cv2
# import matplotlib.pyplot as plt
# from ready_made_cnn1 import CNN_Classification

from kNN_weld import kNN

from defects import main  # импорт класса


class Main_GUI(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.init_main()
        self.m = main()
        self.db = db
        self.view_records()

    def init_main(self):

        # import py_compile
        # py_compile.compile('defects.py')

        # Булева есть ли дефекты на изображении, по умолчанию есть
        self.bool_defect = True
        # self.obj_main=main()

        # меню
        main_menu = tk.Menu()

        file_menu = tk.Menu()
        file_menu.add_command(label="О программе", command=self.about_info)
        file_menu.add_command(label="Помощь", command=self.help_info)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=root.destroy)

        main_menu.add_cascade(label="Информация", menu=file_menu)
        # main_menu.add_cascade(label="Edit")
        # main_menu.add_cascade(label="View")

        root.config(menu=main_menu)

        # Добавление виджета Notebook (Управление вкладкой)
        tab_control = ttk.Notebook(root)
        self.tab1 = ttk.Frame(tab_control)
        self.tab2 = ttk.Frame(tab_control)
        tab_control.add(self.tab1, text='Основная')
        tab_control.add(self.tab2, text='База данных')
        # lbl1 = tk.Label(self.tab1, text='Вкладка 1', padx=5, pady=5)  # Label(tab1, text= 'label1', padx=5, pady=5)
        # lbl1.grid(column=0, row=0)
        # lbl2 = tk.Label(tab2, text='Вкладка 2', padx=5, pady=5)
        # lbl2.grid(column=0, row=0)
        tab_control.pack(expand=1, fill='both')

        # выделенная область
        toolbar = tk.Frame(bg='#d7d8e0', bd=2)
        toolbar.pack(side=tk.TOP)

        # panel1 = tk.Label(toolbar, padx="150", pady="60")
        # panel1.pack(side=tk.LEFT)

        # метка изображения
        label = tk.Label(self.tab1, text="Сварной шов:", justify=tk.LEFT, font="Arial 10", bg='#d7d8e0')
        label.place(x=250, y=20)

        # метка Результат
        label4 = tk.Label(self.tab1, text="Результат:", justify=tk.LEFT, font="Arial 10", bg='#d7d8e0')
        label4.place(x=250, y=280)

        # метка Название детали
        label1 = tk.Label(self.tab1, text="Название детали:", justify=tk.LEFT, font="Arial 10", bg='#d7d8e0')
        label1.place(x=30, y=80)



        # метка дата
        label2 = tk.Label(self.tab1, text="Сегодня: ", justify=tk.LEFT, font="Arial 10", bg='#d7d8e0')
        label2.place(x=30, y=142)

        self.today = date.today()
        str_data = str(self.today.day) + "." + str(self.today.month) + "." + str(self.today.year) + " год"
        # print(str_data)
        label3 = tk.Label(self.tab1, text=str_data, justify=tk.LEFT, font="Arial 10")
        label3.place(x=90, y=142)

        # контейнер
        message = tk.StringVar()
        self.name_weld = ttk.Entry(self.tab1, textvariable=message, font="Arial 10", width=21)
        self.name_weld.place(x=30, y=105)

        # message_button = tk.Button(text="Click Me", command=show_message)
        # message_button.place(relx=.5, rely=.5, anchor="c")

        # кнопка выбрать
        button_img = tk.Button(self.tab1, text="Выбрать изображение",  # текст кнопки  text="Hello",
                               background="#555",  # фоновый цвет кнопки
                               foreground="#ccc",  # цвет текста
                               padx="5",  # отступ от границ до содержимого по горизонтали
                               pady="3",  # отступ от границ до содержимого по вертикали
                               font="Arial 10",  # высота шрифта
                               command=self.insert_img  # действие при нажатии
                               )
        # Чтобы сделать элемент видимым, у него вызывается метод pack()
        button_img.place(x=30, y=20)

        # кнопка поиск
        button_work = tk.Button(self.tab1, text="Поиск дефектов", background="#555", foreground="#ccc", padx="25",
                                pady="3", font="Arial 10", command=self.search_for_defects)
        button_work.place(x=30, y=180)

        # кнопка сохранить
        button_save = tk.Button(self.tab1, text="Сохранить изображение", background="#555", foreground="#ccc", padx="1",
                                pady="3", font="Arial 10", command=self.extract_img)
        button_save.place(x=30, y=230)

        ###########################################################################
        # метка Дефекты
        label10 = tk.Label(self.tab1, text="Дефекты:", justify=tk.LEFT, font="Arial 10", bg='#d7d8e0')
        label10.place(x=30, y=305)

        # фиолетовый # 128, 0, 128
        button1 = tk.Button(self.tab1, text="", background="#800080", foreground="#800080", padx="1",
                                pady="1", font="Arial 10")
        button1.place(x=30, y=335)
        # метка Поры
        label11 = tk.Label(self.tab1, text="- Поры.", justify=tk.LEFT, font="Arial 10")
        label11.place(x=45, y=339)

        # зеленый  0, 128, 0
        button2 = tk.Button(self.tab1, text="", background="#008000", foreground="#008000", padx="1",
                            pady="1", font="Arial 10")
        button2.place(x=30, y=375)
        # метка Шлаковые включения
        label12 = tk.Label(self.tab1, text="- Шлаковые включения.", justify=tk.LEFT, font="Arial 10")
        label12.place(x=45, y=379)

        # синий  0, 0, 255
        button3 = tk.Button(self.tab1, text="", background="#0000FF", foreground="#0000FF", padx="1",
                            pady="1", font="Arial 10")
        button3.place(x=30, y=415)
        # метка Продольные трещины
        label13 = tk.Label(self.tab1, text="- Продольные трещины", justify=tk.LEFT, font="Arial 10")
        label13.place(x=45, y=419)

        # красный   255, 0, 0  #FF0000
        button4 = tk.Button(self.tab1, text="", background="#FF0000", foreground="#FF0000", padx="1",
                            pady="1", font="Arial 10")
        button4.place(x=30, y=455)
        # метка Поперечные трещины
        label14 = tk.Label(self.tab1, text="- Поперечные трещины.", justify=tk.LEFT, font="Arial 10")
        label14.place(x=45, y=459)

        # желтый  255, 255, 0
        button5 = tk.Button(self.tab1, text="", background="#FFFF00", foreground="#FFFF00", padx="1",
                            pady="1", font="Arial 10")
        button5.place(x=30, y=495)
        # метка Без дефектов
        label15 = tk.Label(self.tab1, text="- Без дефекта.", justify=tk.LEFT, font="Arial 10")
        label15.place(x=45, y=499)
        ###########################################################################


        # окно базы данных
        self.open_database()

    def help_info(self):

        root1 = tk.Tk()
        root1.title("Помощь")
        root1.geometry("545x260")
        root1.resizable(False, False)

        help = "Инструкция обработки изображения:\n " \
               "  1. Нажмите \"Выбрать изображение\" и в окне диалога \n" \
               "  выберите изображение формата \".jpg, .jpeg, .png\".\n " \
               "  2. Заполните \"Название детали\".\n " \
               "  3. Нажмите на  \"Поиск дефектов\", чтобы обработать изображение.\n " \
               "  4. Чтобы сохранить результаты нажмите \"Сохранить изображение\" и  \n" \
               "  в окне диалога выберите папку для сохранения.\n\n" \
               "Функции работы c базой данных:\n" \
               "  1. Чтобы изменить название детали, сначала выберите запись из таблицы, \n" \
               "  заполните \"Название детали\", затем нажмите на кнопку \"Изменить наименование\".\n" \
               "  2. Чтобы удалить запись, сначала выберите запись из таблицы,\n" \
               "  затем нажмите на кнопку \"Удалить\".\n"

        label7 = tk.Label(root1, text=help, justify=tk.LEFT, font="Arial 10", bg='#d7d8e0')
        label7.place(x=15, y=15)

    def about_info(self):

        root2 = tk.Tk()
        root2.title("О программе")
        root2.geometry("440x120")
        root2.resizable(False, False)

        about = "Данная программы была разработана к дипломной работе по теме:\n" \
               "\"Анализ дефектов сварных швов с помощью нейронной сети\". \n" \
               "Разработал студент: Назаров Р. М.\n" \
               "Группа: 4414.\n " \

        label8 = tk.Label(root2, text=about, justify=tk.LEFT, font="Arial 10", bg='#d7d8e0')
        label8.place(x=15, y=15)


    # функция выбора изображения
    def insert_img(self):
        # Создает диалоговое окно выбора файла и возвращает имя выбранного файла
        self.file_name = fd.askopenfilename(filetypes=(("jpeg files", "* .jpg"), ("all files", "* . * ")))
        # print(self.file_name)

        # обращаемся главному классу (путь к изоб, False- проверка дефектов, не классифицируем)
        # возвращает, есть ли дефекты(T F)  и изображение
        self.bool_defect, img, self.class_of_defects = self.m.main_function(self.file_name, False)

        pilImage = Image.open(self.file_name)

        self.image1 = self.prepare_output_img(pilImage)
        panel = tk.Label(self.tab1, image=self.image1)
        panel.place(x=250, y=45)

    # подготовка изображения на интерфейс
    def prepare_output_img(self, pilImage):
        h, w = pilImage.size
        # коэффицент
        # если длинну уменьшаем
        scale_h = 1
        scale_w = 1

        if (h > 920):
            scale_h = 920 / max(h, w)
            scale_w = scale_h
            if (w * scale_h > 200):
                scale_w = 200 / w
                scale_h = scale_w
        elif (w > 200):
            scale_w = 200 / w
            scale_h = scale_w

        img = pilImage.resize((int(h * scale_h), int(w * scale_w)), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(img)
        return image

    # кнопка поиск
    def search_for_defects(self):
        # если заранее определили что дефектов нет, то сообщаем это

        if (self.bool_defect == False):
            # label5.delete(0, 'end')

            # метка Дефектов не обнаружено:
            label5 = tk.Label(self.tab1, text="Дефектов не обнаружено.", justify=tk.LEFT, font="Arial 10")
            label5.place(x=320, y=280)

            panel = tk.Label(self.tab1, image=self.image1)
            panel.place(x=250, y=305)
        # если дефекты есть
        else:
            # поиск и классификация(True) дефектов, получаем bool и готовое изображение в массиве numpy
            self.bool_defect, self.img_array, self.class_of_defects = self.m.main_function(self.file_name, True)
            # перевод массива numpy в изображение
            img_ready = Image.fromarray(self.img_array, 'RGB')

            # метка Дефекты
            label5 = tk.Label(self.tab1, text="Дефекты обнаружены.       ", justify=tk.LEFT, font="Arial 10")
            label5.place(x=320, y=280)
            # изображение приводим в порядок для вывода
            self.img_ready = self.prepare_output_img(img_ready)

            panel = tk.Label(self.tab1, image=self.img_ready)
            panel.place(x=250, y=305)

    # команда кнопки сохранить изображение
    def extract_img(self):

        # проверяем пустой или нет
        if len(self.name_weld.get()) == 0:  # empty!
            messagebox.showinfo("Weld", "Заполните название детали!")
        else:
            file_name_save = fd.asksaveasfilename(filetypes=(("jpeg files", "* .jpg"), ("all files", "* . * ")))

            print(file_name_save)
            Image.fromarray(self.img_array, 'RGB').save(file_name_save + ".jpg")

            file_name_out = file_name_save + ".jpg"

            # Список классов
            self.list_defect = ['Без дефекта', 'Трещина продольная', 'Пора', 'Шлаковые включения',
                            'Трещина поперечная']

            self.list_def_db=[]

            temp = np.unique(self.class_of_defects)
            for i in temp:
                self.list_def_db.append(self.list_defect[i])
                #self.list_def_db=  self.list_def_db + self.list_defect[i]

            print(self.list_def_db)
            # self.db.insert_data(description, data, defects, path_img, path_p_img)
            # отправляем данные для записи
            self.db.insert_data(self.name_weld.get(), self.today, str(self.list_def_db), self.file_name, file_name_out)

            # отображение в таблице,обновление
            self.view_records()
            # удаляем содержимое
            self.name_weld.delete(0, 'end')

            # вывод об успешном сохранении
            messagebox.showinfo("Weld", "Успешно сохранено!")

    def open_database(self):
        ##################################
        # height=95 занимаемой области  show='headings'скрыть нулевой
        self.tree = ttk.Treeview(self.tab2,
                                 columns=('ID', 'description', 'data', 'defects', 'original_img', 'processed_img'),
                                 height=15, show='headings')
        self.tree.column("ID", width=30, anchor=tk.CENTER)
        self.tree.column("description", width=250, anchor=tk.CENTER)
        self.tree.column("data", width=120, anchor=tk.CENTER)
        self.tree.column("defects", width=200, anchor=tk.CENTER)
        self.tree.column("original_img", width=200, anchor=tk.CENTER)
        self.tree.column("processed_img", width=200, anchor=tk.CENTER)

        self.tree.heading("ID", text='ID')
        self.tree.heading("description", text='Название детали')
        self.tree.heading("data", text='Дата загрузки')
        self.tree.heading("defects", text='Дефекты')
        self.tree.heading("original_img", text='Фотография оригинал')
        self.tree.heading("processed_img", text='Фотография обработанная')

        self.tree.pack()
        # [0]- первый элемент списка  #1-столбец 1
        # self.tree.selection()[0], '#1')))

        # метка Название детали
        label7 = tk.Label(self.tab2, text="Название детали:", justify=tk.LEFT, font="Arial 10", bg='#d7d8e0')
        label7.place(x=30, y=355)

        # контейнер
        message = tk.StringVar()
        self.rename_weld1 = ttk.Entry(self.tab2, textvariable=message, font="Arial 10", width=22)
        self.rename_weld1.place(x=30, y=380)

        # кнопка Изменить
        button_change = tk.Button(self.tab2, text="Изменить наименование", background="#555", foreground="#ccc",
                                  padx="1",
                                  pady="3", font="Arial 10", command=self.update_record)
        button_change.place(x=30, y=420)

        # кнопка Удалить
        button_delete = tk.Button(self.tab2, text="Удалить", background="#555", foreground="#ccc", padx="50",
                                  pady="3", font="Arial 10", command=self.delete_records)
        button_delete.place(x=30, y=480)

    # отображение на панели
    def view_records(self):
        # запрос в бд
        self.db.c.execute('''SELECT * FROM welds''')
        # очистка таблицы
        [self.tree.delete(i) for i in self.tree.get_children()]

        # self.db.c.fetchall() читает строку
        # insert() добавляет
        [self.tree.insert('', 'end', values=row) for row in self.db.c.fetchall()]

    # self.name_weld.get()
    def update_record(self):

        if len(self.rename_weld1.get()) == 0:  # empty!
            messagebox.showinfo("Weld", "Заполните название детали!")
        else:
            self.db.c.execute('''UPDATE welds SET description=? WHERE ID=?''',
                              (self.rename_weld1.get(), self.tree.set(self.tree.selection()[0], '#1')))
            self.db.conn.commit()
            self.view_records()
            self.rename_weld1.delete(0, 'end')

    def delete_records(self):
        for selection_item in self.tree.selection():
            self.db.c.execute('''DELETE FROM welds WHERE id=?''', (self.tree.set(selection_item, '#1')))
        self.db.conn.commit()
        self.view_records()
        self.rename_weld1.delete(0, 'end')


class DB:
    def __init__(self):
        # создаем соединение с бд, если не существует создат
        self.conn = sqlite3.connect('data_base_weld.db')

        # курсор, добавлять, изменять, удалять

        self.c = self.conn.cursor()
        self.c.execute(
            '''CREATE TABLE IF NOT EXISTS welds (id integer primary key, description text, data numeric, 
            defects text, original_img text, processed_img text)'''
        )
        self.conn.commit()  # сохранить

    # добавление в таблицу
    def insert_data(self, description, data, defects, path_img, path_p_img):
        self.c.execute('''INSERT INTO welds(description, data, defects,original_img, processed_img) 
        VALUES (?, ?, ?, ?, ?)''', (description, data, defects, path_img, path_p_img))
        self.conn.commit()



# условный конструктор
if __name__ == "__main__":
    root = tk.Tk()  # корневое окно программы
    db = DB()
    app = Main_GUI(root)  # вызываем main
    app.pack()  # упоковка метода
    root.title("Defects detection")
    root.geometry("1200x600")
    root.resizable(False, False)  # запрет измерения окна
    root.mainloop()
