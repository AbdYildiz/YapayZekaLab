# Gerekli kutuphanelerin eklenmesi
import pandas as pd
import SimpSOM as sps
from sklearn.cluster import KMeans
import numpy as np

veri = pd.read_csv("airline-safety.csv")
X = veri.drop("airline", "avail_seat_km_per_week", axis=1)
# Agin olusturulmasi
net = sps.somNet(20, 20, X.values, PBC = True)
# Agin egitilmesi
net.train(0.01, 10000)
# Veri noktalarinin 2 boyutlu bir haritaya gomulmesi ve kumelemenin yapilmasi
hrt = np.array((net.project(X.values)))
kmeans = KMeans(n_clusters = 3, max_iter = 300, random_state = 0)
# Kumeleme sonuclarinin gosterilmesi
y_kmeans = kmeans.fit_predict(hrt)
# Kumelerin etiketlerinin belirlenmesi
veri["kumeler"] = kmeans.labels_
# 1 numarali kumenin degerlerine bakilmasi
print(veri[veri["kumeler"] == 0].head(5))
# 2 numarali kumenin degerlerine bakilmasi
print(veri[veri["kumeler"] == 1].head(5))
# 3 numarali kumenin degerlerine bakilmasi
print(veri[veri["kumeler"] == 2].head(5))

#%%
