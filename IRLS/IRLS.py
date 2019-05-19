#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:02:33 2019

@author: Zane Jakobs

Program to solve for L1 regression predictors
using iteratively reweighted least squares
"""
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
# Author: Zane Jakobs
# description: a class to perform L1 regression
# with iteratively reweighted least square
class IRLS:
    
    #Author: Zane Jakobs
    # param external_data: predictor data input as matrix
    # param yobs: observed values of regressor (variable to predict)
    # param objective: what are we solving for? Supported options 
    # param norm: norm to use in solving for resids
    # include only "L1" right now
    def __init__(self, external_data, yobs,objective = "L1",norm = 1 ):
        if np.ma.size(external_data, 0) != np.ma.size(yobs,0):
            print("Error: lengths are not the same.")
        else:
            self.external_data = external_data
            self.yobs = yobs
            self.objective = objective
            self.p = norm
            
    def update_predictors(self, new_data):
        self.external_data = new_data
        
    def update_observations(self, new_yobs):
        self.yobs = new_yobs
        
        
    #@jit
    def regularize_resids(self,resids):
        for e in range(np.ma.size(resids,0)):
                if resids[e] == 0:
                    resids[e] = 1.0e-4
        return resids
    # Author: Zane Jakobs
    # param b: objective vector
    # param a: matrix
    # param x: current x value in the matrix's preimage
    # return: b - ax; the L1 error
    #@jit
    def L1Resids(self,y,a,beta):
        resids = y-np.matmul(a,beta)
        resids = self.regularize_resids(resids)
        return resids
        
    #@jit
    def LpResids(self,y,a,beta):
        resids = y-np.matmul(a,beta)
        resids = self.regularize_resids(np.power(resids,self.p))
        return resids
        

    # Author: Zane Jakobs
    # param resids: residuals vector
    # return: sum of abs values of resids 
    #@jit
    def loss_func(self,resids):
        tot = 0
        for r in np.nditer(resids):
            tot = tot + abs(r)
        return tot
    #@jit
    def LP_loss_func(self,resids):
        tot = 0
        for r in np.nditer(resids):
            tot = tot + abs(r)**self.p
        return tot

    # Author: Zane Jakobs
    # param resids: vector residuals
    # return matrix whose diagonal elements E[i,i] = 1/resids[i]
    #@jit
    def resid_inv_mat(self, resids):
        #assert none of the residuals are 0
        assert np.all(resids != 0)
        return np.diagflat(np.reciprocal(resids))

    # Author: Zane Jakobs
    # param a: constraint/data matrix
    # param Einv: result of a call to resid_inv_mat
    # return: GLS model matrix with covariance matrix Einv (A^T EInv A)
    #@jit
    def model_matrix(self,a,covmat):
        return np.matmul(np.transpose(a), np.matmul(covmat,a) )


    # Author: Zane Jakobs
    # param c: instance of an IRLS object
    # param b: objective vector
    # param a: matrix
    # param currX: current x value in the matrix's preimage
    # return newX: new best solution
    #@jit
    def update_vector(self,y,obvs, Einv, modelMat):
        rhs = np.matmul(np.transpose(obvs), np.matmul(Einv,y))
        return np.linalg.solve(modelMat,rhs)
    
    #Author: Zane Jakobs
    #param old: x^(t-1)
    #param new: x^t
    #return: sum of squares of differences--that is,
    # sum([new-old]^2))
    #@jit
    def sum_square_difference(self,old,new,p,scale):
        tot = 0.0
        for i in range(p):
            o = old[i]
            n = new[i]
            tot = tot + (o - n)**2
            
        return tot/scale
    
    # Author: Zane Jakobs
    # param obs: observations
    # param pred: predictors
    # param beta: estiamte of beta parameters
    #@jit
    def L1SolutionLoop(self,obs,pred,beta):
        epsilon = self.L1Resids(obs,pred,beta)
        Einv = 0.5*self.resid_inv_mat(epsilon)
        modelMat = self.model_matrix(pred,Einv)
        newBeta = self.update_vector(obs, pred, Einv, modelMat)
        rdict = {1:newBeta,2:self.loss_func(epsilon)}
        return rdict
        
   # @jit
    def LPSolutionLoop(self, obs, pred, beta):
        epsilon = self.LpResids(obs,pred,beta)
        if self.p == 2:
            weights = self.resid_inv_mat(epsilon)
        else:
            weights = self.p *0.5* (self.resid_inv_mat(epsilon))**( self.p - 2 )
        modelMat = self.model_matrix(pred,weights)
        newBeta = self.update_vector(obs,pred,weights,modelMat)
        rdict = {1:newBeta,2:self.LP_loss_func(epsilon), 3:np.diag(weights)}
        return rdict

    #@jit
    def getBestBeta(self,betaList, errList):
        minErr = 1.0e15
        listLen = np.ma.size(errList)
        for n in range(listLen):
            if errList[n] < minErr and errList[n] != 0:
                best = n
        return betaList[best]
    
    # Author: Zane Jakobs
    # return: optimal solution
    #@jit
    def solve(self, tolerance = 1.0e-12):
        print("Fitting ", self.objective, " regression.")
        #predictors
        predictors = self.external_data
        #observations
        obs = self.yobs
        n = np.ma.size(obs,0)
        
        obs = obs.reshape((n,1))
        #print(predictors.shape)
        #print(obs.shape)
        #number of predictors
        p = np.ma.size(predictors, 1)
        #number of datapoints
        #n = np.ma.size(obs,0)
        #initialize beta as a 1xp vector of zeros
        beta = np.zeros((p,1))
        #sum of squared differenes between new and old
        #when this is below tolerance, convergence has been reached
        #initialize to a large value
        newOldDist = 1.0e5
        #maximum iterations
        max_iter = 1.0e4
        num_iter = 0
        converged = False
        #arrays to hold stats by iteration
        betaList = [beta]
        errList = [0.0]
        betaScale = 1#obs[4]
        #keep updating beta until convergence
        while newOldDist > tolerance and num_iter < max_iter:
            if self.objective == "L1":
                newData = self.L1SolutionLoop(obs, predictors, beta)
                newBeta = newData[1]
                errList.append(newData[2])
                betaList.append(beta)
                if num_iter == 1:
                    newOldDist = self.sum_square_difference(beta, newBeta,p,betaScale)
                else:
                    newOldDist = self.sum_square_difference(beta, newBeta,p,betaScale)
                beta = newBeta
                num_iter = num_iter + 1
                
            elif self.objective == "LP":
                if self.p == 2:  
                    tolerance = 1.0e-1#stop after one iteration for least squares
                newData = self.LPSolutionLoop(obs,predictors, beta)
                newBeta = newData[1]
                errList.append(newData[2])
                betaList.append(beta)
                newOldDist = self.sum_square_difference(beta, newBeta,p,betaScale)
                beta = newBeta
                num_iter = num_iter + 1
                
            
            else:
                print("Sorry, no  objective functions other than L1 and LP are currently supported")
                break
            
            
        if num_iter >= max_iter - 1:
            print('Solver failed to converge in ',num_iter ,' iterations')
            beta = self.getBestBeta(betaList, errList[1:])
            
        else:
            converged = True
            
        solution_dict = {'Coefficients':beta, 'Errors':errList[1:], 'Beta_v_Iter' : betaList,
                         'Convergence': converged, 
                         'NumIterations':num_iter}
        return solution_dict

    #Author: Zane Jakobs
    #description: plots loss versus iteration
    def plot_loss_vs_iter(self, solution_dict):
        plt.figure()
        plt.plot(solution_dict["Errors"],'r--')
        plt.ylabel("Loss function")
        plt.xlabel("Iteration")
        plt.show()
    
    def plot_uni_regression_line(self, solution_dict):
        beta = solution_dict["Coefficients"]
        line = np.matmul( self.external_data,beta)
        plt.figure()
        plt.plot(range(np.ma.size(self.yobs)), self.yobs,'bx')
        plt.xlabel("Predictor")
        plt.ylabel("Data")
        plt.plot(range(np.ma.size(self.yobs)), line,'-r')
        plt.show()
        
    
    
    
    
    
    
    
    
    
    
    
