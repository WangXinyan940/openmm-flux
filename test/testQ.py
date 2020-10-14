import numpy as np 

p1 = np.array([0., 0., 0.])
p2 = np.array([1., .1, .3])


def calcC(p):
    p3 = np.array([1., 2., 1.1])
    c = np.sqrt(np.power(p-p3, 2).sum()) - 1.
    return c

def func(p1, p2):
    c1 = 1.0
    c2 = calcC(p2)
    dist = np.sqrt(np.power(p1-p2, 2).sum())
    return c1 * c2 * np.math.erfc(dist) / dist

def gradfunc(p1, p2):
    dist = np.sqrt(np.power(p1-p2, 2).sum())
    dcdx = np.zeros((2,2,3))
    delta = np.eye(3) * 0.000001
    for dim in range(3):
        cp1 = 1.0
        cn1 = 1.0
        cp2 = calcC(p2)
        cn2 = calcC(p2)
        dcdx[0,0,dim] = (cp1 - cn1) / 0.000002
        dcdx[1,0,dim] = (cp2 - cn2) / 0.000002
        cp2 = calcC(p2 + delta[dim])
        cn2 = calcC(p2 - delta[dim])
        dcdx[0,1,dim] = (cp1 - cn1) / 0.000002
        dcdx[1,1,dim] = (cp2 - cn2) / 0.000002

    c1 = 1.0
    c2 = calcC(p2)
    grad = np.zeros((3))
    for dim in range(3):
        grad[dim] += c1 * c2 * (- np.math.erfc(dist) / dist ** 2 - 2. * np.exp(- dist ** 2) / np.sqrt(np.pi) / dist) * (p2[dim] - p1[dim]) / dist
        grad[dim] += dcdx[0,1,dim] * c2 * np.math.erfc(dist) / dist
        grad[dim] += c1 * dcdx[1,1,dim] * np.math.erfc(dist) / dist
    return grad


delta = np.eye(3) * 0.000001

numg = np.zeros((3,))
for dim in range(3):
    epos = func(p1, p2 + delta[dim,:])
    eneg = func(p1, p2 - delta[dim,:])
    numg[dim] = (epos - eneg) / 2. / 0.000001
print(numg)
print(gradfunc(p1, p2))