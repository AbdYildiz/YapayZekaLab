from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.models import load_model

# model = Sequential()
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
# serialize weights to HDF5
model = load_model('seydiModel.h5')
print("Saved model to disk")

# image_path = "/Users/abdtester/Documents/Others/la bandera.jpg"
image_path = "/Users/abdtester/Downloads/0027.jpg"
image = load_img(image_path, target_size=(img_size, img_size))
input_arr = img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
input_arr = input_arr.astype('float32') / 255.  # This is VERY important
predictions = model.predict(input_arr)
predicted_class = np.argmax(predictions, axis=-1)
print(predictions)

#%%
