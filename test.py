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

c = 0
for i in test:
    c+=1
    print(f'{c}) Кол-во эпох: {i[0]}, кол-во фильтров: {i[1]}, Кол-во нейронов: {i[2]}, процент случайно сброшенных нейронов: {i[3]}, шаг обучения: {i[4]}')