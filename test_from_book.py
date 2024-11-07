import os
import re
import time
import random
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
# import scikitplot as skplt
from IPython.display import display, Markdown
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.compat import v1 as tf_compat_v1
from sklearn import preprocessing, metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Задание начальных значений и настройка seed для повторяемости
attacks = ['Normal', 'SYN_Flooding', 'ACK_Flooding', 'Port_Scanning', 'OS_Version_Detection',
           'HTTP_Flooding', 'Telnet_Bruteforce', 'UDP_Flooding']
packets = r'B:\Programm\python\lab_bai\lab2\perceptron\packet'
new_packets = r'B:\Programm\python\lab_bai\lab2\perceptron\packets_new'
seed_value = 0

# Установка seed для воспроизводимости
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf_compat_v1.set_random_seed(seed_value)

# Настройка сессии TensorFlow
session_conf = tf_compat_v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf_compat_v1.Session(graph=tf_compat_v1.get_default_graph(), config=session_conf)

# Функции обработки csv
def num_pack_attack(data: pd.DataFrame) -> list:
    cat_count = []
    for cat in set(data.category):
        cat_count.append([cat, (data.category == cat).astype(int).sum()])
    for cat in cat_count:
        print(cat[0], ':', cat[1])
    print('Total packages :', len(data))
    type_of_attack = [cat[0] for cat in cat_count]
    return type_of_attack


def load_dataset_1(packets: str):
    dfs = []
    with os.scandir(packets) as it:
        for entry in it:
            if entry.name.endswith('.csv') and entry.is_file():
                dfs.append(pd.read_csv(entry.path))
    data = pd.concat(dfs)
    data = data.replace(['Infinity', 'NaN', np.inf, -np.inf], np.nan)
    print()

    data = data.drop(columns=[c for c in data.columns if c.endswith('.sum')])
    data = data.drop(columns=[c for c in data.columns if 'port' in c and 'std' not in c])
    data = data.where(pd.notna(data['proto'])).dropna(how='all')
    data.index = range(len(data))
    data = data.fillna(0.0)
    return data, type_of_attack

def load_dataset(packets: str):
    dfs = []
    with os.scandir(packets) as it:
        for entry in it:
            if entry.name.endswith('.csv') and entry.is_file():
                cat = re.findall(r'-([^-]*?).csv', entry.name)[0]
                if cat not in attacks or (cat == 'Normal' and 'benign' not in entry.name):
                    continue
                dfs.append(pd.read_csv(entry.path))
                dfs[-1] = dfs[-1].join(pd.DataFrame(np.full([len(dfs[-1])], cat), columns=['category']))
    data = pd.concat(dfs)
    data = data.replace(['Infinity', 'NaN', np.inf, -np.inf], np.nan)
    type_of_attack = num_pack_attack(data)
    print()

    data = data.drop(columns=[c for c in data.columns if c.endswith('.sum')])
    data = data.drop(columns=[c for c in data.columns if 'port' in c and 'std' not in c])
    data = data.where(pd.notna(data['proto'])).dropna(how='all')
    data.index = range(len(data))
    data = data.fillna(0.0)
    return data, type_of_attack

data, type_attack = load_dataset_1(packets)
data = shuffle(data, random_state=seed_value)

def df_modify(df):
    columns_to_drop = ['srcport.std', 'dstport.std', 'ip.checksum.status.std', 'ip.checksum.status.max',
                       'ip.checksum.status.min', 'ip.checksum.status.mean', 'l4.checksum.status.std',
                       'proto', 'payload.print.mean', 'payload.std.mean', 'payload.mean.mean',
                       'tcp.flags.reset.mean', 'tcp.flags.urg.mean', 'tcp.flags.ecn.mean',
                       'tcp.flags.push.mean', 'tcp.flags.ns.mean', 'tcp.flags.cwr.mean',
                       'tcp.flags.res.mean', 'tcp.flags.fin.mean', 'frame.len.std', 'ip.ttl.std',
                       'int.std', 'ip.flags.rb.mean', 'ip.flags.mf.mean']
    df.drop(columns=columns_to_drop, inplace=True)
    return df

data = df_modify(data)
display(data.columns)

mean_count = int(np.array([sum(data['category'] == cat) for cat in type_attack]).mean())
for cat in type_attack:
    if sum(data['category'] == cat) < mean_count * 0.001:
        data = data.drop(index=data.loc[data['category'] == cat, :].index)
    elif sum(data['category'] == cat) > mean_count:
        data = data.drop(index=data.loc[data['category'] == cat, :][mean_count:].index)

type_attack = num_pack_attack(data)

train_data, test_data, train_labels, test_labels = train_test_split(
    data.drop(columns=['category']), data['category'], test_size=0.3, shuffle=False, random_state=seed_value
)

label_encoder = preprocessing.LabelEncoder()
_ = label_encoder.fit(type_attack)
train_labels = label_encoder.transform(train_labels)
test_labels = label_encoder.transform(test_labels)

data_scaler = preprocessing.StandardScaler()
_ = data_scaler.fit(train_data)

def prepare(data: pd.DataFrame, scaler):
    return scaler.transform(data)

def define_metrics(y_test, y_predicted, y_predicted_proba):
    precision = metrics.precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    recall = metrics.recall_score(y_test, y_predicted, pos_label=None, average='weighted')
    f1 = metrics.f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    matthews_corrcoef = metrics.matthews_corrcoef(y_test, y_predicted)
    try:
        roc_auc_ovr = metrics.roc_auc_score(y_test, y_predicted_proba, average='weighted', multi_class='ovr')
    except ValueError:
        roc_auc_ovr = '-'
    try:
        log_loss = metrics.log_loss(y_test, y_predicted_proba)
    except ValueError:
        log_loss = '-'
    return ['Accuracy', 'Precision', 'Recall', 'F-score', 'MСС', 'ROC AUC', 'Log Loss'], \
           [accuracy, precision, recall, f1, matthews_corrcoef, roc_auc_ovr, log_loss]

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if np.isnan(cm[i, j]):
            cm[i, j] = 0.0
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=12)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=14)
    plt.tight_layout()
    plt.ylabel('True', fontsize=20)
    plt.xlabel('Predicted', fontsize=20)
    return plt

def merge(y_predicted_proba, y_test):
    norm_ind = list(label_encoder.transform(['Normal']))[0]
    y_test_2 = [0 if x == norm_ind else 1 for x in y_test]
    y_predicted_counts_2 = [[x[norm_ind], sum(x) - x[norm_ind]] for x in y_predicted_proba]
    return y_predicted_counts_2, y_test_2

def print_stats(y_predicted_counts_proba_, y_test_, encoder, atk_norm=False):
    if atk_norm:
        y_predicted_counts_proba, y_test = merge(y_predicted_counts_proba_, y_test_)
    else:
        y_predicted_counts_proba, y_test = y_predicted_counts_proba_, y_test_
    y_predicted_counts = np.array(y_predicted_counts_proba).argmax(axis=-1)
    keys, vals = define_metrics(y_test, y_predicted_counts, y_predicted_counts_proba)
    stats = pd.DataFrame([[round(x, 3) if not type(x) == str else x for x in vals]], columns=keys, index=[''])
    display(stats)
    fig_scale = 0.6 if atk_norm else 0.9
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[int(fig_scale * x) for x in (16, 7)])
    _ = skplt.metrics.plot_roc(y_test, y_predicted_counts_proba, figsize=(int(14*0.7), int(10*0.7)),
                               plot_micro=False, plot_macro=False, ax=axs[0])
    if atk_norm:
        cm = metrics.confusion_matrix(y_test, y_predicted_counts)
        _ = plot_confusion_matrix(cm, ['Normal', 'Attack'], normalize=True, title='Confusion matrix')
    else:
        cm = metrics.confusion_matrix(y_test, y_predicted_counts)
        _ = plot_confusion_matrix(cm, encoder.inverse_transform(list(set(y_test))), normalize=True, title='Confusion matrix')
    plt.show()
    return stats

TIME = list()

def t_start() -> float:
    return time.process_time()

def t_end(msg: str, start: float) -> None:
    global TIME
    interval = time.process_time() - start
    print('{:32} : {:2.3f} s'.format(msg, interval))
    TIME.append([msg, interval])

def build_perceptron_model(in_dim: int, out_dim: int):
    model = keras.Sequential()
    model.add(keras.layers.Dense(512, activation='tanh', input_dim=in_dim))
    model.add(keras.layers.Dense(136, activation='sigmoid'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(out_dim, activation='softmax'))
    adam = keras.optimizers.legacy.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Модель перцептрона
perceptron_model = build_perceptron_model(len(train_data.columns), len(label_encoder.classes_))
try:
    t = t_start()
    per = perceptron_model.fit(
        prepare(train_data, data_scaler), train_labels,
        epochs=100, batch_size=128, callbacks=[early_stop]
    )
finally:
    t_end('Multilayer Perceptron Training Time', t)

# Оценка модели
t = t_start()
perceptron_test_predicts = perceptron_model.predict(prepare(test_data, data_scaler), batch_size=128)
t_end('Multilayer Perceptron Runtime', t)
perceptron_model.evaluate(prepare(train_data, data_scaler), train_labels)

# Статистика
bin_stats = list()
multy_stats = list()
t = t_start()
if isinstance(perceptron_model, keras.models.Sequential):
    test_predicts_new = perceptron_model.predict(prepare(test_data, data_scaler))
else:
    test_predicts_new = perceptron_model.predict_proba(prepare(test_data, data_scaler))
t_end('Multilayer Perceptron Predict', t)
bin_stats.append([print_stats(test_predicts_new, test_labels, label_encoder, atk_norm=True)])
multy_stats.append([print_stats(test_predicts_new, test_labels, label_encoder, atk_norm=False)])

# Загрузка нового набора данных
data_new, type_attack_new = load_dataset(new_packets)
for cat in type_attack_new:
    if cat not in type_attack:
        data_new = data_new.drop(index=data_new.loc[data_new['category'] == cat, :].index)
type_attack_new = type_attack
data_new = shuffle(data_new, random_state=seed_value)

mean_count = int(np.array([sum(data_new['category'] == cat) for cat in type_attack_new]).mean())
for cat in type_attack:
    if sum(data_new['category'] == cat) > mean_count:
        data_new = data_new.drop(index=data_new.loc[data_new['category'] == cat, :][mean_count:].index)

type_attack_new = num_pack_attack(data_new)
test_data_new, test_labels_new = data_new.drop(columns=['category']), data_new['category']
test_labels_new = label_encoder.transform(test_labels_new)
test_data_new = test_data_new[train_data.columns]

# Компиляция и тестирование новой модели
perceptron_model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
t = t_start()
if isinstance(perceptron_model, keras.models.Sequential):
    test_predicts_new = perceptron_model.predict(prepare(test_data_new, data_scaler))
else:
    test_predicts_new = perceptron_model.predict_proba(prepare(test_data_new, data_scaler))
t_end('Multilayer Perceptron Predict', t)

# Сохранение статистики для бинарной и многоклассовой модели
bin_stats.append([print_stats(test_predicts_new, test_labels_new, label_encoder, atk_norm=True)])
multy_stats.append([print_stats(test_predicts_new, test_labels_new, label_encoder, atk_norm=False)])
