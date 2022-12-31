# Yaprak siniflandirmasi (1DESA)

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# test ve egitim verilerinin okunmasi
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Siniflarin belirlenmesi ve etiketlenmesi
label_encoder = LabelEncoder().fit(train.species)
labels = label_encoder.transform(train.species)
classes = list(label_encoder.classes_)

# verilerin hazirlanmasi, ozellik ve sinif sayisinin belirlenmesi
train = train.drop(["id", "species"], axis=1)
test = test.drop(["id"], axis=1)
nb_features = 192
nb_classes = len(classes)

# egitim verisindeki verilen standartlastirilirmasi
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train.values)
train = scaler.transform(train.values)

# egitim verisinin egitim ve dogrulama icin ayarlanmasi
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train, labels, test_size=0.1)

# etiketlerin kategorilerinin belirlenmesi
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

# giris verilerinin boyutlarinin ayarlanmasi
x_train = np.array(x_train).reshape(891, 192, 1)
x_valid = np.array(x_valid).reshape(99, 192, 1)

# 1DESA modelinin olusturulmasi
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, MaxPooling1D, Flatten

model = Sequential()
model.add(Conv1D(512, 1, input_shape=(nb_features, 1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()

# agin derlenmesi
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# modelin egitilmesi
model.fit(x_train, y_train, epochs=10, validation_data=(x_valid, y_valid))

# ortalama degerlerin gosterilmesi
print("Ortalama Egitim Kaybi: ", np.mean(model.history.history["loss"]))
print("Ortalama Egitim Basarimi: ", np.mean(model.history.history["accuracy"]))
print("Ortalama Dogrulama Kaybi: ", np.mean(model.history.history["val_loss"]))
print("Ortalama Dogrulama Basarim: ", np.mean(model.history.history["val_accuracy"]))

dtc = DecisionTreeClassifier()
y_pred = dtc.predict(x_valid)
score = f1_score(y_pred, y_valid)
print("F1 score: {:.2f}".format(score))
#%%
