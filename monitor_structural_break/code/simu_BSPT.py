# %%
import timeit
import Bsquid
from operator import mul
import scipy.stats
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
os.chdir("/home/thorondor/Dropbox/Research/MntStrBrk/code")
#%%

pd.options.display.float_format = '{:.2f}'.format

rho = 0.01

pmDef = {'mu1_p_l': 0.5,
         'mu1_p_u': 2.0,
         'mu1_n_l': -2,
         'mu1_n_u': -0.5,
         'mu1_sp_size': 50,
         'rho': rho,
         'smpl_sp_l': -4,
         'smpl_sp_u': 4,
         'smpl_sp_size': 200
         }
pstmf = Bsquid.getPS_TMF(pmDef)

simParam = [[110, 0.8, 0.005],
            [150, 0.8, 0.005],
            [200, 0.8, 0.005],
            [300, 0.8, 0.005],
            [110, 0.8, 0.008],
            [150, 0.8, 0.008],
            [200, 0.8, 0.008],
            [300, 0.8, 0.008],
            [110, 0.8, 0.01],
            [150, 0.8, 0.01],
            [200, 0.8, 0.01],
            [300, 0.8, 0.01]]


def getSimu(cn, tau, alpha, cf):
    smpl = 900
    np.random.seed(cn+3218653)
    y = np.empty(shape=[smpl, ])
    # for i in range(smpl):
    y = np.random.normal(0, 1, smpl)
    y[tau:] = y[tau:] + alpha
    b = {'runStr': 99,
         'rho': rho,
         'pi0': [0.5, 0.5],
         'p21': 0.00,
         'c': cf,
         'pstmf': pstmf,
         'vec': y}
    r = Bsquid.Bsquid(b)
    return(r)


def getBSPT(c):
    pm = simParam[c]
    tau = pm[0]
    alpha = pm[1]
    cf = pm[2]
    nSim = 10000
    df = pd.DataFrame(columns=['c', 'mu1', 'pi', 'piStar', 'y'])
    for i in range(nSim):
        k = i+c
        df.loc[i] = getSimu(k, tau, alpha, cf)
    #df = pd.DataFrame(result, columns=['c', 'mu1', 'pi', 'piStar', 'y'])
    d = df['c'].dropna()
    typeII = (nSim-d.size)/nSim
    typeI = d[d < tau].size/nSim
    ed = np.mean(d[d >= tau]) - tau
    ed_sd = np.std(d[d >= tau])
    return([tau, alpha, cf, ed, ed_sd, typeI, typeII])


pool = mp.Pool(15)
result = np.array(pool.map(getBSPT, range(len(simParam))))
pool.close()
pool.join()

result = pd.DataFrame(result,
                      columns=['tau', 'alpha', 'cf', 'expected depay', 'sd', 'type I', 'type II'])

result.to_csv("../output/results.csv", sep=',', index=False)
print("done")


# %%
