import math as mt
import random as rd

rd.seed(1)

#neural function
def RN(m1,m2):
    t = m1*w1+m2*w2+b
    return sigmoide(t)

def sigmoide(t):
    return 1/(1+mt.exp(-t))

def sigmoide_p(t):
    return sigmoide(t)*(1- sigmoide(t))

#our dataset
dataset =  [[6.3,   5,      1],
            [7.1,   6.3,    0],
            [7,     6,      0],
            [5.9,   4.5,    1],
            [6.8,   5.8,    0],
            [6,     4.6,    1],
            [5.5,   4.2,    1],
            [6.9,   5.8,    0]]

#tarining procedure
def train():
    w1 = rd.random() 
    w2 = rd.random() 
    b = rd.random()

    iterarions = 10000
    learning_rate = 0.1

    for i in range(iterarions):
        ri = rd.randint(0,len(dataset)-2)
        point = dataset[ri] 

        z = point[0] * w1 + (point[1]) * w2 + b
        pred = sigmoide(z)

        target = point[2]

        cost = (pred - target)**2
        dcost_dpred = 2 * (pred - target) 
        dpred_dz = sigmoide_p(z)
        dz_dw1 = point[0]
        dz_dw2 = point[1]
        dz_db = 1 

        dcost_dz = dcost_dpred * dpred_dz

        dcost_dw1 = dcost_dz * dz_dw1
        dcost_dw2 = dcost_dz * dz_dw2
        dcost_db = dcost_dz * dz_db

        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db
    
    return w1, w2, b

w1, w2, b = train()

#testing
pred = []
for elephant in dataset:
    z = w1 * elephant[0] + w2 * elephant[1] + b
    prediction = sigmoide(z)
    if(prediction >= 0.5):
        pred.append('indian ' + str(prediction) + "\n")
    else:
        pred.append('african ' + str(prediction) + "\n")

print(pred)
