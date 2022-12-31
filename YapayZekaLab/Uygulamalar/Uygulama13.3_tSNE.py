#  170541028 ABDULLAH YILDIZ

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("digit-recognizer/train.csv")
test = pd.read_csv("digit-recognizer/test.csv")

label_encoder = LabelEncoder().fit(train.label)
labels = label_encoder.transform(train.label)
classes = list(label_encoder.classes_)

train = train.drop(["label"], axis=1)
nb_features = 784
nb_classes = len(classes)

standardized_data = StandardScaler().fit_transform(train)
sample_data = standardized_data
covar_matrix = np.matmul(sample_data.T , sample_data)
from scipy.linalg import eigh
values, vectors = eigh(covar_matrix, eigvals=(782, 783))
vector = vectors.T
vector[[0,1]]=vector[[1,0]]

import seaborn as sn
import matplotlib.pyplot as plt
new_coordinates = np.matmul(vector, sample_data.T)

new_coordinates = np.vstack((new_coordinates, labels)).T
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))

# TSNE
from sklearn.manifold import TSNE
# Picking the top 1000 points as TSNE takes a lot of time for 15K points
data_1000 = standardized_data[0:1000,:]
labels_1000 = labels[0:1000]
model = TSNE(n_components=2, random_state=0)
# configuring the parameteres
# the number of components = 2
tsne_data = model.fit_transform(data_1000)
# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

#%%
