# leaf classification
# https://www.kaggle.com/c/leaf-classification/data
# 3DESA ile goruntu siniflandirma
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

print(os.listdir("leaf-classification"))
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# egitim verisinin hazirlanmasi
# filenames = os.listdir("leaf-classification/images")
# categories = []
# for filename in filenames:
# 	category = filename.split('.')[0]
# 	if category == 'dog':  # 1 numarali sinif kopek
# 		categories.append(1)
# 	else:
# 		categories.append(0)  # 0 numarali sinif kopek
# df = pd.DataFrame({'filename': filenames, 'category': categories})

# df['category'].value_counts().plot.bar()
# image = load_img("leaf-classification/images/"+random.choice(filenames))
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
model.add(Dense(2, activation='softmax'))  # 2 because we have cat and dog classes
model.summary()

#  modelin derlenmesi
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# verinin hazirlanmasi
df["category"] = df["category"].replace({0: "cat", 1: "dog"})
train_df, validate_df = train_test_split(df, test_size=0.2)
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
	rotation_range=15,
	rescale=1. / 255,
	shear_range=0.1,
	zoom_range=0.2,
	horizontal_flip=True,
	width_shift_range=0.1,
	height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
	train_df,
	"CNN/train",
	x_col='filename',
	y_col='category',
	target_size=IMAGE_SIZE,
	class_mode='categorical',
	batch_size=batch_size
)

# dogrulama verilerinin cogaltilmasi
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
	validate_df,
	"CNN/train",
	x_col='filename',
	y_col='category',
	target_size=IMAGE_SIZE,
	class_mode='categorical',
	batch_size=batch_size
)

# modelin egitilmesi
epochs = 10
history = model.fit(
	train_generator,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=total_validate // batch_size,
	steps_per_epoch=total_train // batch_size
)

# olusturulan modelin kaydedilmesi
model.save_weights("model1.h5")
# tahmin isleminin yapilmasi
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples / batch_size))
# %%
