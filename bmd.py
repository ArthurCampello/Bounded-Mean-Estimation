# this file contains functions to produce posterior mean distributions using 
# MBC-IA methods of Uniform and Jeffreys priors and CLT and student-t methods
# for data in [0,1]. These methods are described in the attached manuscript

# needed imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t
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

# returns mean posterior distribution from data x (input vector 'x') in [0,1]
# from the MBC-IA with a uniform prior, as described in the manuscript, with count vectors of lengths n (input int 'n')
@jit(nopython=True)
def mdist_up(x, n):
    dc = data2counts(x, n)
    res = 2**n-1
    con = np.array([beta(dc[0][1]+1,dc[0][0]+1,j) for j in np.linspace(0,1,int(2**(n-1)))])
    for i in range(1,n-1):
        con = conv(con,np.array([beta(dc[i][1]+1,dc[i][0]+1,j) for j in np.linspace(0,1,int(2**(n-i-1))+1)]))
    return np.linspace(0,1,len(con)), con/np.sum(con)

# returns mean posterior distribution from data x (input vector 'x') in [0,1]
# from the MBC-IA with a Jeffreys prior, as described in the manuscript, with count vectors of lengths n (input int 'n')
@jit(nopython=True)
def mdist_jp(x, n):
    dc = data2counts(x, n)
    res = 2**n-1
    con = np.array([beta(dc[0][1]+1/2,dc[0][0]+1/2,j) for j in np.linspace(0,1,int(2**(n-1)))])
    for i in range(1,n-1):
        con = conv(con,np.array([beta(dc[i][1]+1/2,dc[i][0]+1/2,j) for j in np.linspace(0,1,int(2**(n-i-1))+1)]))
    return np.linspace(0,1,len(con)), con

# returns mean posterior distribution from data x (input vector 'x') in [0,1]
# from the CLT method, as described in the manuscript, with count vectors of lengths n (input int 'n')
def mdist_cl(d, n):
    m = 2**n-2
    x = np.linspace(0,1,m)
    return x, norm.pdf(x, np.mean(d), np.std(d)/np.sqrt(len(d))) 

# returns mean posterior distribution from data x (input vector 'x') in [0,1]
# from the student-t method, as described in the manuscript, with count vectors of lengths n (input int 'n')
def mdist_st(d, n):
    m = 2**n-2
    x = np.linspace(0,1,m)
    return x, t.pdf(x, len(d)-1, np.mean(d), np.std(d)/np.sqrt(len(d)))
