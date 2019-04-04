#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Original code
Author: Andre Pacheco
Email: pacheco.comp@gmail.com

Generalized TODIM code
Author: Leandro Lima

This class implements a generalized TODIM [4] algorithm, based in the TODIM [1,2] algorithm.
In order to use it, you need to inform the decision matrix, criteria's weights and thetas's value.
You can set these parameters in an external file .txt or just call the constructors passing 
the variables as parameters.

Additionally, one can set their own f, g, distance and comparison functions. Default funtions are those used in [3] for TODIM.

In the file task_todim.py there is an example showing how to use this class.

For more information about TODIM:
    [1] L.F.A.M. Gomes, M.M.P.P. Lima TODIM: Basics and application to multicriteria ranking of projects with environmental impacts
        Foundations of Computing and Decision Sciences, 16 (4) (1992), pp. 113-127
    
    [2] Krohling, Renato A., Andre GC Pacheco, and Andre LT Siviero. IF-TODIM: An intuitionistic fuzzy TODIM to multi-criteria decision
        making. Knowledge-Based Systems 53, (2013), pp. 142-146.

    [3] Lourenzutti, R. and Khroling, R. A study of TODIM in a intuitionistic fuzzy and random environment,
        Expert Systems with Applications, Expert Systems with Applications 40, (2013), pp. 6459-6468

    [4] Llamazares, Bonifacio. An analysis of the generalized TODIM method. European Journal of Operational Research 269, (2018), pp. 1041–1049.


If you find any bug, please e-mail me =)

'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from FuzzyNumber import FuzzyNumber

class TODIM:
    '''
    Attributes:
    matrixD - The decision matrix with the alternatives and criteria
    weights - The weights for each criteria
    theta - The theta's value
    nAlt - The number of alternatives
    nCri - The number of criteria
    normMatrixD - The matrixD normalized
    rCloseness - The relative closeness coeficient    
    f = f[1] = f_1 phi function and f[2] = f_2 phi function, default are the simplified TODIM functions
    g = g[1] = g_1 phi function and g[2] = g_2 phi function, default are the simplified TODIM functions
    distance = distance function, default is the euclidian distance
    comparison = compare function, default is the difference
    '''
    matrixD = None
    weights = None
    wref = None
    theta = None
    nAlt = None
    nCri = None
    normMatrixD = None
    phi = None
    delta = None
    rCloseness = None
    f = None 
    g = None
    distance = None
    comparison = None
    
    # TODIM(filename)
    # TODIM(matrixD, weights, theta)
    def __init__ (self, *args):
        nargs = len(args)        
        if nargs == 0:
            print ('ERROR: There is no parameter in the construction function')
            raise ValueError
        elif nargs == 1 or nargs == 2:
            # The .txt file need to be the 1st parameter
            fileName = args[0]
            try:
                data = np.loadtxt(fileName, dtype=float)
            except IOError:
                print ('ERROR: there is a problem with the .txt file. Please, check it again')
                raise IOError

            # All the values are in the .txt
            if nargs == 1:
                self.weights = data[0,:]
                self.setTheta( data[1,0])
                self.matrixD = data[2:,:]                
            # Only the matrixD is passed in the .txt
            else:
                self.matrixD = data
                self.weights = np.asarray(args[0])
                self.setTheta (args[1])
        # In this case, all the parameters are passed without use a .txt, in the following order: matrixD, weights, theta
        elif nargs == 3:
            self.matrixD = np.asarray(args[0])
            self.weights = np.asarray(args[1])
            self.setTheta( args[2] )
               
        #Just checking if the weights' sum is equals 1
        weights_sum = self.weights.sum()
        if round( weights_sum , 5) != 1:
            self.weights = self.weights/weights_sum            
            print  ('The weights was normalized in the interval [0,1]')
            
                        
        # Filling the remaining variables
        size = self.matrixD.shape
        [self.nAlt, self.nCri] = size
        self.normMatrixD = np.zeros(size, dtype=float)           
        self.delta = np.zeros([self.nAlt, self.nCri])
        self.rCloseness = np.zeros ([self.nAlt,1], dtype=float)
        # weight reference
        self.wref = self.weights.max()
        
        # Default f and g functions definition based in [3] (equation in section 3) definition
        def f1_simplified_TODIM (self, x):
            return np.sqrt(x)
        
        def f2_simplified_TODIM (self, x):
            return np.sqrt(x)
        

        def g1_simplified_TODIM (self, x):
            return np.sqrt(x)
        
        def g2_simplified_TODIM (self, x):
            return np.sqrt(x)/ self.getTheta()

        self.setF (f1_simplified_TODIM, f2_simplified_TODIM)
        self.setG (g1_simplified_TODIM, g2_simplified_TODIM)
    
        def euclidianDistance (a, b):
            return abs(a - b)

        self.setDistance (euclidianDistance)
        
        def differenceComparison (a, b):
            return a - b

        self.setComparison (differenceComparison)



    def printTODIM (self):      
        print ('MatrixD \n', self.matrixD)
        print ('Weights \n', self.weights)
        print ('Theta \n', self.theta)

    # Normalizeing the matrixD
    def normalizeMatrix (self):
        m = self.matrixD.sum(axis=0)
        for i in range(self.nAlt):
            for j in range(self.nCri):
                self.normMatrixD[i,j] = self.matrixD[i,j] / m[j]
    
        self.matrixD = self.normMatrixD

    # theta setter
    def setTheta (self, v_theta):
        self.theta = v_theta

    def getTheta (self):
        return self.theta

    # g_1 and g_2 setter
    # Must be g_1, g_2: (0,1) -> (0,+inf)
    def setG (self, g1, g2):
        self.g = [0, g1, g2]

    # f_1 and f_2 setter
    # Must be f_1, f_2: [0,1] -> [0,+inf) AND f_1(0) = f_2(0) = 0
    def setF (self, f1, f2):
        self.f = [0, f1, f2]

    # distance function setter
    # Must be distance >= 0
    def setDistance (self, dist_function):
        self.distance = dist_function

    def calcDistance (self, a, b):
        return self.distance(a,b)
    
    # comparison function
    def setComparison (self, compare_function):
        self.comparison = compare_function

    
    # make comparison based in previous defined function
    def getComparison (self, a, b):        
        return self.comparison(a, b)

    def getDelta (self):
        for i in range(self.nAlt):
            for j in range(self.nCri):
                self.delta[i,j] = self.getSumPhi(i,j)

                
    def getSumPhi (self,i,j):
        #m = np.zeros(self.nCri)
        m = 0
        for k in range(self.nCri):
            m = m + self.getPhi(i,j,k)
        return m
    
    ##
    #
    #                |  g_1(w_k) * f_1(Z_ik - Z_jk)  , if Z_ik >= Z_jk
    # phi_k(Ai,Aj) = |
    #                | -g_2(w_k) * f_1(Z_jk - Z_ik)  , if Z_ik <  Z_jk 
    #
    ##
    def getPhi (self, i, j, k):
        Z_ik = self.matrixD[i, k]
        Z_jk = self.matrixD[j, k]

        dij = self.calcDistance(Z_ik, Z_jk)
        comp = self.getComparison (Z_ik, Z_jk)
        if comp == 0:
            return 0
        elif comp > 0:
            return self.g[1](self, self.weights[k]) * self.f[1](self, dij)
        else:
            return -self.g[2](self, self.weights[k]) * self.f[2](self, dij)

    def getRCloseness (self, verbose=False, name=None):
        self.getDelta()
        aux = self.delta.sum(axis=1)
        for i in range(self.nAlt):
            self.rCloseness[i] = (aux[i] - aux.min()) / (aux.max() - aux.min())
        if verbose:
            return (self.rCloseness)
            
    # To plot the Alternatives' name, just pass a list of names
    # To save the plot, just pass the files name on saveName
    def plotBars (self,names=None, saveName=None):        
        sns.set_style("whitegrid")
        if names is not None:
            a = sns.barplot (names, self.rCloseness[:,0], palette="BuGn_d")
        else:
            a = sns.barplot (None, self.rCloseness[:,0], palette="BuGn_d")
        
        a.set_ylabel("Closeness Coeficient")
        a.set_xlabel('Alternatives')
        fig = a.get_figure()
        plt.show()
        
        
        if saveName is not None:
            fig.savefig(saveName+'.png')

################################## END CLASS ####################################################

