# coding: utf-8

import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np

# 各年代の画像をリスト化
Apple_files = os.listdir('downloads/Apple')
Tomato_files = os.listdir('downloads/Tomato')

# 配列Xに画像を入れる
X = []
for i in Apple_files:
    img = cv2.imread('downloads/Apple/'+i)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, dsize=(100, 100))
    X.append(rgb)

for i in Tomato_files:
    img = cv2.imread('downloads/Tomato/'+i)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, dsize=(100, 100))
    X.append(rgb)

# Yに各ラベルを入れる
Y = []
label_names = ['Apple', 'Tomato']
for i in range(len(Apple_files)):
    Y.append(0)

for i in range(len(Tomato_files)):
    Y.append(1)


# 顔画像全表示
plt.figure(figsize=(10, 10))
for i in range(len(X)):
    plt.subplot(10, 40, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i], cmap=plt.cm.binary)
    plt.xlabel(label_names[Y[i]])

plt.show()

# 正規化
for i in range(len(X)):
    X[i] = X[i]/255


# 学習用とテスト用
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = np.array(X_train)
X_train = X_train.reshape(-1, 100, 100, 3)
X_test = np.array(X_test)
X_test = X_test.reshape(-1, 100, 100, 3)
input_shape = X_train.shape[1:]


# モデル構築
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習実行
model.fit(X_train, Y_train, epochs=30)
model.save('AppleTomato_model.h5')

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
predictions = model.predict(X_test)

# 正解不正解画像表示定義


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(label_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         label_names[true_label]),
               color=color)

# 正解不正解ラベル表示定義


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 結果表示
num_rows = 2
num_cols = 2
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, Y_test, X_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, Y_test)
plt.show()
