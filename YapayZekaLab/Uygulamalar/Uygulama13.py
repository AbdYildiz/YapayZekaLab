#  170541028 ABDULLAH YILDIZ

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

label_encoder = LabelEncoder().fit(train.species)
labels = label_encoder.transform(train.species)
classes = list(label_encoder.classes_)

train = train.drop(["id", "species"], axis=1)
test = test.drop(["id"], axis=1)
nb_features = 192
nb_classes = len(classes)

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train, labels, test_size=0.2)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

x_train = np.array(x_train).reshape(792, 192, 1)
x_valid = np.array(x_valid).reshape(198, 192, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten, LSTM, BatchNormalization

model = Sequential()
model.add(LSTM(512, input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

from keras import backend as k

def recall_m(y_true, y_pred):
	true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
	possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + k.epsilon())
	return recall

def precision_m(y_true, y_pred):
	true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
	predicted_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
	precision = true_positives / (predicted_positives + k.epsilon())
	return precision

def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true,y_pred)
	return 2* ((precision*recall) / (precision+recall+k.epsilon()))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1_m, precision_m, recall_m])
score = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid))

print("Ortalama Egitim Kaybi : ", np.mean(model.history.history["loss"]))
print("Ortalama Egitim Basarimi : ", np.mean(model.history.history["accuracy"]))
print("Ortalama Dogrulama Kaybi : ", np.mean(model.history.history["val_loss"]))
print("Ortalama Dogrulama Basarimi : ", np.mean(model.history.history["val_accuracy"]))
print("Ortalama F1-Skor Degeri : ", np.mean(model.history.history["val_f1_m"]))
print("Ortalama Kesinlik Degeri : ", np.mean(model.history.history["val_precision_m"]))
print("Ortalama Duyarlik Degeri : ", np.mean(model.history.history["val_recall_m"]))

import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Basarimlari")
plt.ylabel("Basarim")
plt.xlabel("Epok Sayisi")
plt.legend(["Egitim", "Dogrulama"], loc="upper left")
plt.show()

plt.plot(model.history.history["loss"], color="g")
plt.plot(model.history.history["val_loss"], color="r")
plt.title("Model Kayiplari")
plt.ylabel("Kayip")
plt.xlabel("Epok Sayisi")
plt.legend(["Egitim", "Dogrulama"], loc="upper left")
plt.show()

plt.plot(model.history.history["f1_m"], color="y")
plt.plot(model.history.history["val_f1_m"], color="b")
plt.title("Model F1-Skorlari")
plt.ylabel("F1-Skor")
plt.xlabel("Epok Sayisi")
plt.legend(["Egitim", "Dogrulama"], loc="upper left")
plt.show()
#%%
