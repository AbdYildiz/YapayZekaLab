import numpy as np

a = np.array([5,8,11])
b = np.array([[5,8,11],[6,9,45]])
#creating 3x3 zeros matrix
zerosMatrix = np.zeros([3,3])
#sqrt and log for a given array
squareRoot = np.sqrt(a)
logarithm = np.log(a)
#average and transpose for a given array
ortalama = np.mean(a)
transpoz = np.transpose(b)
#deleting matrix element
elemanCikar = np.delete(b,[3])

print(elemanCikar)
print(ortalama)
print(transpoz)
