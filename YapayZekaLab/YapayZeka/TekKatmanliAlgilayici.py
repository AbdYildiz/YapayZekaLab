X1 = [1, 0]
X2 = [0, 1]
W = [1, 2]
B = [1, 0]
teta = -1
lamda = 0.5


def net(X, W):
    net = 0
    for i in range(len(X)):
        net += W[i] * X[i]
    print("net : ", net)
    return net


def out(X, W):
    if net(X, W) > teta:
        return 1
    else:
        return 0


def iter(X, W, B):
    Exp = out(X, W)
    print("W : ", W)
    if Exp != B:
        if Exp == 1:
            W[0] = W[0] - lamda * X[0]
            W[1] = W[1] - lamda * X[1]
        else:
            W[0] = W[0] + lamda * X[0]
            W[1] = W[1] + lamda * X[1]
    else:
        return B


while True:
    if iter(X1, W, B[0]) == B[0] and iter(X2, W, B[1]) == B[1]:
        break
    else:
        iter(X1, W, B[0])
        iter(X2, W, B[1])

#%%
