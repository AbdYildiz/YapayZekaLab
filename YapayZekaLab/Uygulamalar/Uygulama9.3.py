# leaf
# https://www.kaggle.com/competitions/leaf-classification
# 3DESA ile goruntu siniflandirma
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from sklearn.preprocessing import LabelEncoder

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


# print(os.listdir("leaf-classification"))
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# egitim verisinin hazirlanmasi
filenames = os.listdir("CNN/train")
categories = []
for filename in filenames:
	category = filename.split('.')[0]
	if category == 'dog':  # 1 numarali sinif kopek
		categories.append(1)
	else:
		categories.append(0)  # 0 numarali sinif kopek
df = pd.DataFrame({'filename': filenames, 'category': categories})

df['category'].value_counts().plot.bar()
# image = load_img("CNN/train/"+random.choice(filenames))
# plt.imshow(image)

# modelin olusturulmasi
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))  # nb_classes tane sinif
model.summary()

# agin derlenmesi##########################
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#  modelin derlenmesi
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# verinin hazirlanmasi
# df["category"] = df["category"].replace({0:"cat", 1:"dog"})
train_df, validate_df = train_test_split(df, test_size= 0.2)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
# kategorilere bakilmasi
train_df["category"].value_counts().plot.bar()
# egitim ve dogrulama verisinin hazirlanmasi
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

#  egitim verilerinin cogaltilmasi
train_datagen = ImageDataGenerator(
	rotation_range = 15,
	rescale = 1./255,
	shear_range = 0.1,
	zoom_range = 0.2,
	horizontal_flip = True,
	width_shift_range = 0.1,
	height_shift_range = 0.1
)

train_generator = train_datagen.flow_from_dataframe(
	train_df,
	"leaf-classification/images",
	x_col = 'filename',
	y_col = 'category',
	target_size=IMAGE_SIZE,
	class_mode='categorical',
	batch_size=batch_size
)

# dogrulama verilerinin cogaltilmasi
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator =validation_datagen.flow_from_dataframe(
	validate_df,
	"leaf-classification/images",
	x_col = 'filename',
	y_col = 'category',
	target_size=IMAGE_SIZE,
	class_mode='categorical',
	batch_size=batch_size
)

# modelin egitilmesi #########################################
# model.fit(x_train, y_train, epochs=15, validation_data=(x_valid, y_valid))

# modelin egitilmesi
epochs = 15  # 10
history = model.fit(
	train_generator,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=total_validate//batch_size,
	steps_per_epoch=total_train//batch_size
)

# olusturulan modelin kaydedilmesi
# model.save_weights("model1.h5")
# tahmin isleminin yapilmasi
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
#%%
