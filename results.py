
import numpy as np
import os

epochs = [50]
# alphas = [0.0001, 0.001, 0.00001, 0.00005]
alphas = [0.0001]
# scales = [2, 3, 4, 5, 6, 7, 8, 9, 10]
scales = [8, 16, 32]
lows = [0.01, 0.005, 0.001]

'''
epochs = [1]
alphas = [0.0001]
scales = [2]
'''

for epoch in epochs:
    for alpha in alphas:
        for scale in scales:
            for low in lows:
                cmd = "python mnist.py --epochs %d --alpha %f --scale %f --low %f" % (epoch, alpha, scale, low)
                os.system(cmd)

for epoch in epochs:
    for alpha in alphas:
        for scale in scales:
            for low in lows:
                name = "./results/epochs_%d_alpha_%f_scale_%f_low_%f.npy"% (epoch, alpha, scale, low)
                acc = np.load(name)
                acc = np.max(acc)
                print ("epochs %d alpha %f scale %f low %f acc %f" % (epoch, alpha, scale, low, acc))
