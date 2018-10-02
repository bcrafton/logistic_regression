
import numpy as np
import os

epochs = [10]
alphas = [0.0005, 0.001, 0.005]
scales = [2, 4, 8, 16, 32]
lows = np.linspace(0.005, 0.1, 20)
# lows = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005]

for epoch in epochs:
    for scale in scales:
        for low in lows:
            for alpha in alphas:
                cmd = "python mnist.py --epochs %d --alpha %f --scale %f --low %f" % (epoch, alpha, scale, low)
                os.system(cmd)

table = []

for epoch in epochs:
    for scale in scales:
        for low in lows:
            for alpha in alphas:
                name = "./results/epochs_%d_alpha_%f_scale_%f_low_%f.npy"% (epoch, alpha, scale, low)
                acc = np.load(name)
                acc = np.max(acc)
                table.append( (epoch, alpha, scale, low, acc) )
                print ("epochs %d alpha %f scale %f low %f acc %f" % (epoch, alpha, scale, low, acc))
     

header = "epochs,alpha,scale,low,acc"
np.savetxt(fname='results.csv', X=table, delimiter=',', header=header)
