import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veri = pd.read_csv("telefon_fiyat_degisimi.csv")

# Sinif sayisi ve etiketinin belirlenmesi
label_encoder = LabelEncoder().fit(veri.price_range)
labels = label_encoder.transform(veri.price_range)
classes = list(label_encoder.classes_)

x = veri.drop(["price_range"], axis=1)
x = x.drop(["blue"], axis=1)
x = x.drop(["fc"], axis=1)
x = x.drop(["int_memory"], axis=1)
x = x.drop(["ram"], axis=1)
x = x.drop(["wifi"], axis=1)
y = labels
nb_features = 15
nb_classes = len(classes)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# girdi verilerinin yeniden boyutlandirilmasi
x_train = np.array(x_train).reshape(1400, 15, 1)
x_test = np.array(x_test).reshape(600, 15, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Flatten, SimpleRNN, BatchNormalization

model = Sequential()
model.add(SimpleRNN(512, input_shape=(nb_features, 1)))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="sigmoid"))
model.summary()

from tensorflow.keras.optimizers import SGD
opt = SGD(learning_rate= 1e-3, decay= 1e-5, momentum= 0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer= opt, metrics=["accuracy"])

score = model.fit(x_train, y_train, epochs=250, validation_data=(x_test, y_test))

print("Ortalama Egitim Kaybi: ", np.mean(model.history.history["loss"]))
print("Ortalama Egitim Basarimi: ", np.mean(model.history.history["accuracy"]))
print("Ortalama Dogrulama Kaybi: ", np.mean(model.history.history["val_loss"]))
print("Ortalama Dogrulama Basarimi: ", np.mean(model.history.history["val_accuracy"]))

import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Basarimlari")
plt.ylabel("Basarim")
plt.xlabel("Epok sayisi")
plt.legend(["Egitim", "Test"], loc="upper left")
plt.show()

plt.plot(model.history.history["loss"], color="g")
plt.plot(model.history.history["val_loss"], color="r")
plt.title("Model Kayiplari")
plt.ylabel("Kayip")
plt.xlabel("Epok sayisi")
plt.legend(["Egitim", "Test"], loc="upper left")
plt.show()

#%%
