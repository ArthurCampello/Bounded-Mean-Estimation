# this file contains functions to produce posterior mean distributions using 
# MBC-BC methods of Uniform and Jeffreys priors for data in [0,1]. 
# These methods are described in the attached manuscript

# needed imports
import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from numba import jit

# pdf of beta distribution
# takes in beta distribution parameters 'a' and 'b' and a value 'x'
# in [0,1] and returns beta distribution pdf at that value
@jit(nopython=True)
def beta(a,b,x):
    if x <= 0 or x>=1:
        return 0
    else:
        return x**(a-1)*(1-x)**(b-1)/((gamma(a)+gamma(b))/gamma(a+b))
    
# takes in two vectors 'fa' and 'fb' and returns their normalized convolution
@jit(nopython=True)
def conv(fa,fb):
    r = np.convolve(fa,fb)
    return r/np.sum(r)

# Transforms data X (input vector 'x') in [0,1] into k-vector 
# of length n (input int 'n') as described in manuscript
@jit(nopython=True)
def data2counts(x, n):
    if min(x) < 0 or max(x) > 1:
      raise Exception("Values must be in [0,1]")
    res = np.zeros((n,2))
    for i in x:
        for j in range(n):
            m = int(2**(j+1)*i % 2)
            res[j][m] = res[j][m] + 1
    return res

# multinomial pobability function
@jit(nopython=True)
def mnom(xb, h1, h2, h3):    
    k = np.array([h1+h2+h3-1, h1-h2-h3+1, -h1+h2-h3+1, -h1-h2+h3+1])/2
    return np.prod(np.array([k[i]**xb[i] for i in range(4)]))

# envelope function
@jit(nopython=True)
def envelope(x, y):
    return [1-np.abs(y-x),np.abs(x+y-1)]

# calculates bivariate distribution of theta1 and theta2 under uniform prior assumptions
@jit(nopython=True)
def bivariatedist_up(x, n):
    xb = -0.5*np.ones(4)
    for i in x:
        if 0 <= i < 0.25:
            xb[2] = xb[2] + 1
        elif 0.25 <= i < 0.5:
            xb[3] = xb[3] + 1
        elif 0.5 <= i < 0.75:
            xb[0] = xb[0] + 1
        else:
            xb[1] = xb[1] + 1
    dist = np.zeros((n,n))
    delt = 0.0001
    for j in range(n):
        for k in range(0,n,16):
            env = envelope((j+0.5)/n, (k+0.5)/n)           
            dom = env[0]-env[1]
            spoints = np.linspace(env[1]+delt, env[0]-delt, int(dom*20)+1)           
            dist[j][k] = np.mean(np.array([mnom(xb,(j+0.5)/n,(k+0.5)/n,l) for l in spoints]))*dom                    
    return dist/np.sum(dist)

# calculates bivariate distribution of theta1 and theta2 under Jeffreys prior assumptions
@jit(nopython=True)
def bivariatedist_jp(x, n):
    xb = -0.75*np.ones(4)
    for i in x:
        if 0 <= i < 0.25:
            xb[2] = xb[2] + 1
        elif 0.25 <= i < 0.5:
            xb[3] = xb[3] + 1
        elif 0.5 <= i < 0.75:
            xb[0] = xb[0] + 1
        else:
            xb[1] = xb[1] + 1
    dist = np.zeros((n,n))
    delt = 0.0001
    for j in range(n):
        for k in range(0,n,16):
            env = envelope((j+0.5)/n, (k+0.5)/n)           
            dom = env[0]-env[1]
            spoints = np.linspace(env[1]+delt, env[0]-delt, int(dom*20)+1)           
            dist[j][k] = np.mean(np.array([mnom(xb,(j+0.5)/n,(k+0.5)/n,l) for l in spoints]))*dom                    
    return dist/np.sum(dist)

# calculates distribution of theta1+theta2 from bivariate distribution matrix 'biv'
@jit(nopython=True)
def distfrombiv(biv):
    s = biv.shape
    res = np.zeros(int(s[1]*3/2))    
    biv = np.vstack((biv, np.zeros((int(s[0]/2),s[0]))))
    biv = np.vstack((np.zeros((int(s[0]/2),s[0])), biv))
    for i in range(0,int(s[1]*3/2)):
        for j in range(0,s[0],16):
            res[i] = res[i] + biv[int(i+j/2)][j] 
    return res/np.sum(res)

# returns mean posterior distribution from data x (input vector 'x') in [0,1]
# from the MBC-BC with a Jeffreys prior, as described in the manuscript, with count vectors of lengths n (input int 'n')
@jit(nopython=True)
def mdist_biv_up(x, n):
    dc = data2counts(x, n)
    res = 2**n-1
    con = distfrombiv(bivariatedist_up(x, 512))
    for i in range(2,n-1):
        con = conv(con,np.array([beta(dc[i][1]+1,dc[i][0]+1,j) for j in np.linspace(0,1,int(2**(n-i-1))+1)]))
    return np.linspace(0,1,len(con)), con/np.sum(con)

# returns mean posterior distribution from data x (input vector 'x') in [0,1]
# from the MBC-BC with a Jeffreys prior, as described in the manuscript, with count vectors of lengths n (input int 'n')
@jit(nopython=True)
def mdist_biv_jp(x, n):
    dc = data2counts(x, n)
    res = 2**n-1
    con = distfrombiv(bivariatedist_jp(x, 512))
    for i in range(2,n-1):
        con = conv(con,np.array([beta(dc[i][1]+1/2,dc[i][0]+1/2,j) for j in np.linspace(0,1,int(2**(n-i-1))+1)]))
    return np.linspace(0,1,len(con)), con/np.sum(con)
