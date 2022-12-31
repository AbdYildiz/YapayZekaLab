import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.datasets.fashion_mnist import load_data as tf

veri = pd.read_csv("aerial-cactus-identification/train.csv")
test = pd.read_csv("aerial-cactus-identification/sample_submission.csv")

# Sinif sayisi ve etiketinin belirlenmesi
label_encoder = LabelEncoder().fit(veri.has_cactus)
labels = label_encoder.transform(veri.has_cactus)
classes = list(label_encoder.classes_)

x = veri.drop(["has_cactus"], axis=1)
y = labels
nb_features = 1
nb_classes = len(classes)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1/255.0)
train_dir = "aerial-cactus-identification/train/"
batch_size = 64
image_size = 32
veri.has_cactus = veri.has_cactus.astype(str)
x_train = datagen.flow_from_dataframe(dataframe=veri[:14000],directory=train_dir,x_col='id',
                                              y_col='has_cactus',class_mode='binary',batch_size=batch_size,
                                              target_size=(image_size,image_size))


x_test = datagen.flow_from_dataframe(dataframe=veri[14000:],directory=train_dir,x_col='id',
                                                   y_col='has_cactus',class_mode='binary',batch_size=batch_size,
                                                   target_size=(image_size,image_size))

test_dir = "aerial-cactus-identification/test/"
batch_size = 64
image_size = 32
test.has_cactus = test.has_cactus.astype(str)
y_train = datagen.flow_from_dataframe(dataframe=test[:2000],directory=test_dir,x_col='id',
                                      y_col='has_cactus',batch_size=batch_size,
                                      target_size=(image_size,image_size))


y_test = datagen.flow_from_dataframe(dataframe=test[2000:],directory=test_dir,x_col='id',
                                     y_col='has_cactus',batch_size=batch_size,
                                     target_size=(image_size,image_size))

# girdi verilerinin yeniden boyutlandirilmasi
# x_train = np.array(x_train).reshape(12250, 1, 1)
# x_test = np.array(x_test).reshape(5250, 1, 1)

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

#%%
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