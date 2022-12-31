import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

train_path = 'dataset/train'
valid_path = 'dataset/valid'

def PlotImage(img_arr):
	fig, axes = plt.subplots(1, 5, figsize=(20, 20))
	axes = axes.flatten()
	for img, ax in zip(img_arr, axes):
		ax.imshow(img)
	plt.tight_layout()
	plt.show()


training_datagen = ImageDataGenerator(rescale=1 / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                      shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

training_data = training_datagen.flow_from_directory(train_path, target_size=(200, 200), batch_size=128,
                                                     class_mode='binary')
training_data.class_indices
valid_datagen = ImageDataGenerator(rescale=1 / 255)

valid_data = training_datagen.flow_from_directory(valid_path, target_size=(200, 200), batch_size=128,
                                                  class_mode='binary')
images = [training_data[0][0][0] for i in range(5)]
PlotImage(images)

model_save_path = 'face_mask_detection_model.h5'
checkpoint = ModelCheckpoint(model_save_path , monitor='val_accuracy' ,verbose=1 , save_best_only=True , mode='max')
callbacks_list = [checkpoint]
model = keras.models.Sequential([
	keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[200, 200, 3]),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Conv2D(filters=64, kernel_size=3),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Conv2D(filters=128, kernel_size=3),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Conv2D(filters=256, kernel_size=3),
	keras.layers.MaxPooling2D(pool_size=(2,2)),

	keras.layers.Dropout(0.5),
	keras.layers.Flatten(), # neural network beulding
	keras.layers.Dense(units=128, activation='relu'), # input layers
	keras.layers.Dropout(0.1),
	keras.layers.Dense(units=256, activation='relu'),
	keras.layers.Dropout(0.25),
	keras.layers.Dense(units=2, activation='softmax') # output layer
])
model.compile(optimizer= 'adam' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
model.summary()

history = model.fit(training_data ,
                    epochs=50 ,
                    verbose=1,
                    validation_data= valid_data , callbacks = callbacks_list )

model.save("face_mask_detection_model.h5")

#%%