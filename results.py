
import numpy as np
import os

epochs = [10]
alphas = [0.001, 0.0001]
scales = [2, 4, 8, 16, 32]
lows = [0.1, 0.05, 0.01, 0.005, 0.001]

for epoch in epochs:
    for alpha in alphas:
        for scale in scales:
            for low in lows:
                cmd = "python mnist.py --epochs %d --alpha %f --scale %f --low %f" % (epoch, alpha, scale, low)
                os.system(cmd)

table = []

for epoch in epochs:
    for alpha in alphas:
        for scale in scales:
            for low in lows:
                name = "./results/epochs_%d_alpha_%f_scale_%f_low_%f.npy"% (epoch, alpha, scale, low)
                acc = np.load(name)
                acc = np.max(acc)
                table.append( (epoch, alpha, scale, low, acc) )
                print ("epochs %d alpha %f scale %f low %f acc %f" % (epoch, alpha, scale, low, acc))
     

header = "epochs,alpha,scale,low,acc"
np.savetxt(fname='results.csv', X=table, delimiter=',', header=header)
