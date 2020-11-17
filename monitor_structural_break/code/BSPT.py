import numpy as np
import pandas as pd
import scipy.stats
from operator import mul
'''
########################################################################
define paramater space and function of transition matrix for mu change
pmDef = {'mu1_p_l':0.6,
         'mu1_p_u':2.0,
         'mu1_n_l':-2,
         'mu1_n_u':-0.6,
         'mu1_sp_size':50,
         'rho':0.01,
         'smpl_sp_l': -4,
         'smpl_sp_u':4,
         'smpl_sp_size': 200
        }
define the BSPT object
brkStr = 109
brkSize = 0.8
smpl = 600
runStr = 99
y = np.random.normal(0,1,smpl)
y[brkStr:] = y[brkStr:] + brkSize
b ={'runStr' : 99,
    'rho':0.01,
    'pi0':[0.5,0.5],
    'p21':0,
    'c':0.01,
    'PS_TMF':PS_TMF
    'vec':y}
##########################################################################
define paramater space and function of transition matrix for sigma change
pmDef_sigma = {'sigma1ul':1.3,
               'sigma1uu':2.0,
               'sigma1ll':0.0,
               'sigma1lu':0.7,
               'sigma1_sp_size':50,
               'rho':0.01,
               'smpl_sp_l': -4,
               'smpl_sp_u':4,
               'smpl_sp_size': 200
              }
define the BSPT object
brkStr = 109
brkSize = 0.8
smpl = 600
runStr = 99
y = np.random.normal(0,1,smpl)
y[brkStr:] = y[brkStr:] + brkSize
b ={'runStr' : 99,
    'rho':0.01,
    'pi0':[0.5,0.5],
    'p21':0,
    'c':0.01,
    'PS_TMF':PS_TMF
    'vec':y}
############################################################################
'''
def getPS_TMF(pmDef):
    mu1Sp = list(np.linspace(pmDef['mu1_n_l'],pmDef['mu1_n_u'],pmDef['mu1_sp_size']))+list(np.linspace(pmDef['mu1_p_l'],pmDef['mu1_p_u'],pmDef['mu1_sp_size']))
    pmSp = pd.DataFrame(mu1Sp,columns=['mu1'])
    pmSp['zero'] = 0
    smplSp = list(np.linspace(pmDef['smpl_sp_l'],pmDef['smpl_sp_u'],pmDef['smpl_sp_size']))
    tmf = [getTM(smplSp,pmDef['rho'],mu1) for mu1 in pmSp['mu1']]
    PS_TMF={'pmSp' : pmSp,
            'tmf'  : tmf }
    return(PS_TMF)

def getTV(rho,df,pi):
    t_tmp = pd.DataFrame(np.arange(0,101)/100,columns=['pi_prime'])
    df['pi_prime'] = np.round((df['lr']*(pi+rho*(1-pi)))/(df['lr']*(pi+rho*(1-pi))+(1-rho)*(1-pi)),2)
    d = df.groupby('pi_prime',as_index=False).sum()
    d['p'] = (1-pi)*(1-rho)*d['p0']+(pi+(1-pi)*rho)*d['p1']
    t_tmp = pd.merge(t_tmp,d[['pi_prime','p']],on='pi_prime',how='left')
    t_tmp = np.array(t_tmp.iloc[:,1].fillna(0))
    return(t_tmp.reshape(1,-1))

def getTM(smplSp,rho,mu1):
    piSp = np.array(range(0,101))/100
    df = pd.DataFrame(np.zeros((len(smplSp),6)),columns=['l0','l1','p0','p1','lr','pi_prime'])
    df['l0'] = scipy.stats.norm(0,1).pdf(smplSp)
    df['l1'] = scipy.stats.norm(mu1,1).pdf(smplSp)
    df['p0'] = df['l0']/sum(df['l0'])
    df['p1'] = df['l1']/sum(df['l1'])
    df['lr'] = df['l1']/df['l0']
    tm = np.zeros([len(piSp),len(piSp)])
    for i in range(len(piSp)): 
        tm[i,] = getTV(rho,df,piSp[i])
    return(tm)

def BSPT(theObject):
    rho = theObject['rho']
    pi0 = theObject['pi0']
    p21 = theObject['p21']
    c = theObject['c']
    y = theObject['vec']
    runStr = theObject['runStr']
    pmSp = theObject['pstmf']['pmSp']
    tmf  = theObject['pstmf']['tmf']
    p = np.array([[1-rho, p21],
                  [rho, 1-p21]])
    opt = pd.DataFrame(columns=['c','mu1_bar',"pi","piStar","y"])
    for i in np.arange(runStr,len(y)):
        # initiate iteration
        if i == runStr:
            y0 = y[0:runStr]
            pmSp_tmp = pd.DataFrame([getLikelihood(pi0,p,p21,mu1,y0) for mu1 in pmSp['mu1']],columns=['mu1','l','piTheta'])
            pmSp['l'] = pmSp_tmp['l']
            pmSp['piTheta'] = pmSp_tmp['piTheta']
            pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            opt.loc[i] = [i,sum(pmSp.mu1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),0,0]
        else:
            x = y[i]
            pi0_1 = sum(pmSp.piTheta*pmSp.posd)
            pi0 = np.array([1-pi0_1,pi0_1])
            l = np.array([scipy.stats.norm(pmSp.zero,1).pdf(x),scipy.stats.norm(pmSp.mu1,1).pdf(x)])
            k = p@pi0
            pmSp['f']=k@l
            pmSp['piTheta'] = k[1]*l[1,:]/pmSp['f']
            pmSp['f'] = (pmSp['f'])/sum(pmSp['f'])
            pmSp['posd'] = (pmSp['f']*pmSp['posd'])/sum(pmSp['f']*pmSp['posd'])
            # get last period's posterior as current period's prior
            # this step will update the sum of log likelihood, so in the pmSp the l will always be the sum of log
            # likelihood for all available data
            #pmSp.loc[:,'l'] = pmSp['l'] + np.log(pmSp.f)
            # update the parameter space with new posterior as next period's prior
            #pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            tm = list(map(mul,tmf,pmSp['posd']))
            t = np.array(sum(tm))
            piStar = getPistar(rho,c,t)
            opt.loc[i] = [i,sum(pmSp.mu1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),piStar,x]
            if opt.loc[i,'pi'] > opt.loc[i,'piStar']:
                return([opt.loc[i,'c'],opt.loc[i,'mu1_bar'],opt.loc[i,'pi'],opt.loc[i,'piStar'],x])
                break
            if i == len(y)-1:
                return([np.NaN,opt.loc[i,'mu1_bar'],opt.loc[i,'pi'],opt.loc[i,'piStar'],x])

def BSPT_crtGraph(theObject):
    rho = theObject['rho']
    pi0 = theObject['pi0']
    p21 = theObject['p21']
    c = theObject['c']
    y = theObject['vec']
    runStr = theObject['runStr']
    pmSp = theObject['pstmf']['pmSp']
    tmf  = theObject['pstmf']['tmf']
    p = np.array([[1-rho, p21],
                  [rho, 1-p21]])
    opt = pd.DataFrame(columns=['c','mu1_bar',"pi","piStar","y"])
    for i in np.arange(runStr,len(y)):
        # initiate iteration
        if i == runStr:
            y0 = y[0:runStr]
            pmSp_tmp = pd.DataFrame([getLikelihood(pi0,p,p21,mu1,y0) for mu1 in pmSp['mu1']],columns=['mu1','l','piTheta'])
            pmSp['l'] = pmSp_tmp['l']
            pmSp['piTheta'] = pmSp_tmp['piTheta']
            pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            opt.loc[i] = [i,sum(pmSp.mu1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),0,y[runStr]]
        else:
            x = y[i]
            pi0_1 = sum(pmSp.piTheta*pmSp.posd)
            pi0 = np.array([1-pi0_1,pi0_1])
            l = np.array([scipy.stats.norm(pmSp.zero,1).pdf(x),scipy.stats.norm(pmSp.mu1,1).pdf(x)])
            k = p@pi0
            pmSp['f']=k@l
            pmSp['piTheta'] = k[1]*l[1,:]/pmSp['f']
            pmSp['f'] = (pmSp['f'])/sum(pmSp['f'])
            pmSp['posd'] = (pmSp['f']*pmSp['posd'])/sum(pmSp['f']*pmSp['posd'])
            # get last period's posterior as current period's prior
            # this step will update the sum of log likelihood, so in the pmSp the l will always be the sum of log
            # likelihood for all available data
            #pmSp.loc[:,'l'] = pmSp['l'] + np.log(pmSp.f)
            # update the parameter space with new posterior as next period's prior
            #pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            tm = list(map(mul,tmf,pmSp['posd']))
            t = np.array(sum(tm))
            piStar = getPistar(rho,c,t)
            opt.loc[i] = [i,sum(pmSp.mu1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),piStar,x]
    return(opt)

def BSPT_getPostDist(theObject):
    rho = theObject['rho']
    pi0 = theObject['pi0']
    p21 = theObject['p21']
    c = theObject['c']
    y = theObject['vec']
    runStr = theObject['runStr']
    pmSp = theObject['pstmf']['pmSp']
    postDist = theObject['pstmf']['pmSp']
    tmf  = theObject['pstmf']['tmf']
    p = np.array([[1-rho, p21],
                  [rho, 1-p21]])
    opt = pd.DataFrame(columns=['c','mu1_bar',"pi","piStar","y"])
    for i in np.arange(runStr,len(y)):
        # initiate iteration
        if i == runStr:
            y0 = y[0:runStr]
            pmSp_tmp = pd.DataFrame([getLikelihood(pi0,p,p21,mu1,y0) for mu1 in pmSp['mu1']],columns=['mu1','l','piTheta'])
            pmSp['l'] = pmSp_tmp['l']
            pmSp['piTheta'] = pmSp_tmp['piTheta']
            pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            opt.loc[i] = [i,sum(pmSp.mu1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),0,y[runStr]]
            postDist.loc[:,i] = pmSp['posd']
        else:
            x = y[i]
            pi0_1 = sum(pmSp.piTheta*pmSp.posd)
            pi0 = np.array([1-pi0_1,pi0_1])
            l = np.array([scipy.stats.norm(pmSp.zero,1).pdf(x),scipy.stats.norm(pmSp.mu1,1).pdf(x)])
            k = p@pi0
            pmSp['f']=k@l
            pmSp['piTheta'] = k[1]*l[1,:]/pmSp['f']
            pmSp['f'] = (pmSp['f'])/sum(pmSp['f'])
            pmSp['posd'] = (pmSp['f']*pmSp['posd'])/sum(pmSp['f']*pmSp['posd'])
            postDist.loc[:,i] = pmSp['posd']
            # get last period's posterior as current period's prior
            # this step will update the sum of log likelihood, so in the pmSp the l will always be the sum of log
            # likelihood for all available data
            #pmSp.loc[:,'l'] = pmSp['l'] + np.log(pmSp.f)
            # update the parameter space with new posterior as next period's prior
            #pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            tm = list(map(mul,tmf,pmSp['posd']))
            t = np.array(sum(tm))
            piStar = getPistar(rho,c,t)
            opt.loc[i] = [i,sum(pmSp.mu1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),piStar,x]
    return(postDist)
    
def BSPT_getVF(theObject):
    rho = theObject['rho']
    pi0 = theObject['pi0']
    p21 = theObject['p21']
    c = theObject['c']
    y = theObject['vec']
    runStr = theObject['runStr']
    pmSp = theObject['pstmf']['pmSp']
    tmf  = theObject['pstmf']['tmf']
    p = np.array([[1-rho, p21],
                  [rho, 1-p21]])
    opt = pd.DataFrame(columns=['c','mu1_bar',"pi","piStar","y"])
    for i in np.arange(runStr,130):
        # initiate iteration
        if i == runStr:
            y0 = y[0:runStr]
            pmSp_tmp = pd.DataFrame([getLikelihood(pi0,p,p21,mu1,y0) for mu1 in pmSp['mu1']],columns=['mu1','l','piTheta'])
            pmSp['l'] = pmSp_tmp['l']
            pmSp['piTheta'] = pmSp_tmp['piTheta']
            pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            opt.loc[i] = [i,sum(pmSp.mu1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),0,y[runStr]]
        else:
            x = y[i]
            pi0_1 = sum(pmSp.piTheta*pmSp.posd)
            pi0 = np.array([1-pi0_1,pi0_1])
            l = np.array([scipy.stats.norm(pmSp.zero,1).pdf(x),scipy.stats.norm(pmSp.mu1,1).pdf(x)])
            k = p@pi0
            pmSp['f']=k@l
            pmSp['piTheta'] = k[1]*l[1,:]/pmSp['f']
            pmSp['f'] = (pmSp['f'])/sum(pmSp['f'])
            pmSp['posd'] = (pmSp['f']*pmSp['posd'])/sum(pmSp['f']*pmSp['posd'])
            # get last period's posterior as current period's prior
            # this step will update the sum of log likelihood, so in the pmSp the l will always be the sum of log
            # likelihood for all available data
            #pmSp.loc[:,'l'] = pmSp['l'] + np.log(pmSp.f)
            # update the parameter space with new posterior as next period's prior
            #pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            tm = list(map(mul,tmf,pmSp['posd']))
            t = np.array(sum(tm))
            v = getVF(rho,c,t)
    return(v)
    
def getLikelihood(pi0,p,p21,mu1,y):
    l0 = scipy.stats.norm(0,1).pdf(y)
    l1 = scipy.stats.norm(mu1,1).pdf(y)
    pi = []
    f = []
    for i in range(len(y)):
        if i == 0:
                f.append(np.dot(np.dot(p,pi0),[l0[i],l1[i]]))
                pi.append(np.dot(p,pi0)*[l0[i],l1[i]]/f[i])
        else:
                f.append(np.dot(np.dot(p,pi[i-1]),[l0[i],l1[i]]))
                pi.append(np.dot(p,pi[i-1])*[l0[i],l1[i]]/f[i])
    return([mu1,sum(np.log(f)),pi[len(pi)-1][1]])

def getPS_TMF_sigma(pmDef):
    sigma1Sp = list(np.linspace(pmDef['sigma1ul'],pmDef['sigma1uu'],pmDef['sigma1_sp_size']))+list(np.linspace(pmDef['sigma1ll'],pmDef['sigma1lu'],pmDef['sigma1_sp_size']))
    pmSp = pd.DataFrame(sigma1Sp,columns=['sigma1'])
    pmSp['zero'] = 0
    smplSp = list(np.linspace(pmDef['smpl_sp_l'],pmDef['smpl_sp_u'],pmDef['smpl_sp_size']))
    tmf = [getTM_sigma(smplSp,pmDef['rho'],sigma1) for sigma1 in pmSp['sigma1']]
    PS_TMF={'pmSp' : pmSp,
            'tmf'  : tmf }
    return(PS_TMF)

def getTM_sigma(smplSp,rho,sigma1):
    piSp = np.array(range(0,101))/100
    df = pd.DataFrame(np.zeros((len(smplSp),6)),columns=['l0','l1','p0','p1','lr','pi_prime'])
    df['l0'] = scipy.stats.norm(0,1).pdf(smplSp)
    df['l1'] = scipy.stats.norm(0,sigma1).pdf(smplSp)
    df['p0'] = df['l0']/sum(df['l0'])
    df['p1'] = df['l1']/sum(df['l1'])
    df['lr'] = df['l1']/df['l0']
    tm = np.zeros([len(piSp),len(piSp)])
    for i in range(len(piSp)): 
        tm[i,] = getTV(rho,df,piSp[i])
    return(tm)

def BSPT_sigma(theObject):
    rho = theObject['rho']
    pi0 = theObject['pi0']
    p21 = theObject['p21']
    c = theObject['c']
    y = theObject['vec']
    runStr = theObject['runStr']
    pmSp = theObject['pstmf']['pmSp']
    tmf  = theObject['pstmf']['tmf']
    p = np.array([[1-rho, p21],
                  [rho, 1-p21]])
    opt = pd.DataFrame(columns=['c','sigma1_bar',"pi","piStar","y"])
    for i in np.arange(runStr,len(y)):
        # initiate iteration
        if i == runStr:
            y0 = y[0:runStr]
            pmSp_tmp = pd.DataFrame([getLikelihood_sigma(pi0,p,p21,sigma1,y0) for sigma1 in pmSp['sigma1']],columns=['sigma1','l','piTheta'])
            pmSp['l'] = pmSp_tmp['l']
            pmSp['piTheta'] = pmSp_tmp['piTheta']
            pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            opt.loc[i] = [i,sum(pmSp.sigma1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),0,0]
        else:
            x = y[i]
            pi0_1 = sum(pmSp.piTheta*pmSp.posd)
            pi0 = np.array([1-pi0_1,pi0_1])
            l = np.array([scipy.stats.norm(pmSp.zero,1).pdf(x),scipy.stats.norm(pmSp.zero,pmSp.sigma1).pdf(x)])
            k = p@pi0
            pmSp['f']=k@l
            pmSp['piTheta'] = k[1]*l[1,:]/pmSp['f']
            pmSp['f'] = (pmSp['f'])/sum(pmSp['f'])
            pmSp['posd'] = (pmSp['f']*pmSp['posd'])/sum(pmSp['f']*pmSp['posd'])
            # get last period's posterior as current period's prior
            # this step will update the sum of log likelihood, so in the pmSp the l will always be the sum of log
            # likelihood for all available data
            #pmSp.loc[:,'l'] = pmSp['l'] + np.log(pmSp.f)
            # update the parameter space with new posterior as next period's prior
            #pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            tm = list(map(mul,tmf,pmSp['posd']))
            t = np.array(sum(tm))
            piStar = getPistar(rho,c,t)
            opt.loc[i] = [i,sum(pmSp.sigma1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),piStar,x]
            if opt.loc[i,'pi'] > opt.loc[i,'piStar']:
                return([opt.loc[i,'c'],opt.loc[i,'sigma1_bar'],opt.loc[i,'pi'],opt.loc[i,'piStar'],x])
                break
            if i == len(y):
                return([np.NaN,opt.loc[i,'sigma1_bar'],opt.loc[i,'pi'],opt.loc[i,'piStar'],x])

def BSPT_sigma_crtGraph(theObject):
    rho = theObject['rho']
    pi0 = theObject['pi0']
    p21 = theObject['p21']
    c = theObject['c']
    y = theObject['vec']
    runStr = theObject['runStr']
    pmSp = theObject['pstmf']['pmSp']
    tmf  = theObject['pstmf']['tmf']
    p = np.array([[1-rho, p21],
                  [rho, 1-p21]])
    opt = pd.DataFrame(columns=['c','sigma1_bar',"pi","piStar","y"])
    for i in np.arange(runStr,len(y)):
        # initiate iteration
        if i == runStr:
            y0 = y[0:runStr]
            pmSp_tmp = pd.DataFrame([getLikelihood_sigma(pi0,p,p21,sigma1,y0) for sigma1 in pmSp['sigma1']],columns=['sigma1','l','piTheta'])
            pmSp['l'] = pmSp_tmp['l']
            pmSp['piTheta'] = pmSp_tmp['piTheta']
            pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            opt.loc[i] = [i,sum(pmSp.sigma1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),0,0]
        else:
            x = y[i]
            pi0_1 = sum(pmSp.piTheta*pmSp.posd)
            pi0 = np.array([1-pi0_1,pi0_1])
            l = np.array([scipy.stats.norm(pmSp.zero,1).pdf(x),scipy.stats.norm(pmSp.zero,pmSp.sigma1).pdf(x)])
            k = p@pi0
            pmSp['f']=k@l
            pmSp['piTheta'] = k[1]*l[1,:]/pmSp['f']
            pmSp['f'] = (pmSp['f'])/sum(pmSp['f'])
            pmSp['posd'] = (pmSp['f']*pmSp['posd'])/sum(pmSp['f']*pmSp['posd'])
            # get last period's posterior as current period's prior
            # this step will update the sum of log likelihood, so in the pmSp the l will always be the sum of log
            # likelihood for all available data
            #pmSp.loc[:,'l'] = pmSp['l'] + np.log(pmSp.f)
            # update the parameter space with new posterior as next period's prior
            #pmSp['posd']=np.exp(pmSp['l']-pmSp['l'].mean(axis=0))/sum(np.exp(pmSp['l']-pmSp['l'].mean(axis=0)))
            tm = list(map(mul,tmf,pmSp['posd']))
            t = np.array(sum(tm))
            piStar = getPistar(rho,c,t)
            opt.loc[i] = [i,sum(pmSp.sigma1*pmSp.posd),sum(pmSp.piTheta*pmSp.posd),piStar,x]
    return(opt)
    


def getLikelihood_sigma(pi0,p,p21,sigma1,y):
    l0 = scipy.stats.norm(0,1).pdf(y)
    l1 = scipy.stats.norm(0,sigma1).pdf(y)
    pi = []
    f = []
    for i in range(len(y)):
        if i == 0:
                f.append(np.dot(np.dot(p,pi0),[l0[i],l1[i]]))
                pi.append(np.dot(p,pi0)*[l0[i],l1[i]]/f[i])
        else:
                f.append(np.dot(np.dot(p,pi[i-1]),[l0[i],l1[i]]))
                pi.append(np.dot(p,pi[i-1])*[l0[i],l1[i]]/f[i])
    return([sigma1,sum(np.log(f)),pi[len(pi)-1][1]])

def getPistar(rho,c,t):
    piSp = np.array(range(0,101))/100
    # this iteration method is following Shiryaev's book 4.126-4.128
    # h = c*np.matmul(t,piSp)+1+(c-1)*piSp
    h = c*piSp+1-np.matmul(t,piSp)
    g = 1-piSp
    Q = np.amin([g,h],axis=0)
    ep = 1
    cnt = 0
    while ep > 0.001 and cnt < 500:
        cnt = cnt + 1
        Q1 = Q
        h = np.matmul(t,Q) + c*piSp
        Q = np.amin([g,h],axis=0)
        ep = max(abs(Q1-Q))
        diff = abs(g-h)
        if cnt < 500:
            piStar = piSp[np.argmin(diff)]
        else:
            piStar = 'NaN'
    return(piStar)
    
def getVF(rho,c,t):
    piSp = np.array(range(0,101))/100
    v = pd.DataFrame()
    v[0] = piSp
    # this iteration method is following Shiryaev's book 4.126-4.128
    h = c*piSp+1-np.matmul(t,piSp)
    g = 1-piSp
    Q = np.amin([g,h],axis=0)
    ep = 1
    cnt = 0
    while ep > 0.001 and cnt<500:
        cnt = cnt + 1
        Q1 = Q
        h = np.matmul(t,Q) + c*piSp
        Q = np.amin([g,h],axis=0)
        v[cnt] = h
        ep = max(abs(Q1-Q))
        diff = abs(g-h)
        if cnt < 500:
            piStar = piSp[np.argmin(diff)]
        else:
            piStar = 'NaN'
    return(v)