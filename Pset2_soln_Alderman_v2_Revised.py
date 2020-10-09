# -*- coding: utf-8 -*-
"""
David Alderman
Intro to Econometrics
Problem Set #2
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
print('\n')
# Make changes for Fun!
with open(r'C:\Users\david\OneDrive\Documents\1a. Columbia MSFE\2020.09_Intro_to_Econometrics\psets\Pset 2\nls_2008.txt') as f:
          matrix = [line.split() for line in f]

matrix = np.array(matrix).astype(np.float)

# arrays of each variable
luwe = matrix[:, 0] # log weekly earnings
educ = matrix[:, 1] # years of education
exper = matrix[:, 2] # years of experience
age = matrix[:, 3] # age in years
fed = matrix[:, 4] # Father's education in years
med = matrix[:, 5] # Mother's education in years
kww = matrix[:, 6] # Test score
iq = matrix[:, 7] # IQ Score
white = matrix[:, 8] # Indicator for white

#--------------------------------------------------------
# Question 1
print('Problem 1''\n')
matrix_col = matrix.shape[1] # Number of columns in matrix
sum_matrix = np.zeros(shape = (4,matrix_col))

for i in range(matrix_col): # Calcualting avg, max, min, mean for each column
    avg_c = np.mean(matrix[:,i]) 
    max_c = max(matrix[:,i]) 
    min_c = min(matrix[:,i])
    std_c = np.std(matrix[:,i])
    sum_matrix[0][i] = min_c
    sum_matrix[1][i] = max_c
    sum_matrix[2][i] = avg_c
    sum_matrix[3][i] = std_c
sum_matrix_df = pd.DataFrame(data=np.transpose(sum_matrix), 
                             index=['luwe','educ', 'exper', 'age', 'fed', 'med', 'kww', 'IQ', 'White'],
                             columns=['Min','Max','Mean', 'Std'])
print('Summary Table')
print(sum_matrix_df) 
print('\n')  

#--------------------------------------------------------
# Question 2
# Normal Linear Regression for luwe on educ, experience, and experience squared
# Experience defined as age less education less six

# Model defined as luwe = beta_not + beta_1 * educ + 
    # beta_2 * experience + beta_3 * experience^2 + residual
print('-'*40)
print('\n''Problem 2''\n')
def exper(age, educ):
    experience = age - educ - 6
    return experience

n = luwe.shape[0]
k = 4 # number of parameters

experience = exper(age, educ) # experience as defined in problem statement
exp_sq = np.square(experience) 
data = np.transpose(np.stack((educ,experience,exp_sq)))

residual = luwe - data.dot(np.linalg.inv(np.transpose(data).dot(data))).dot(np.transpose(data).dot(luwe))
residual_var = (np.transpose(residual).dot(residual)) / (n - k - 1)

#### Regression Model Function ########
#######################################

# Standard Error
def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

# Robust Error
def reg_m_r(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit(cov_type='HC1')
    return results
#######################################
#######################################

output = reg_m(np.transpose(luwe),np.transpose(data))
output_robust = reg_m_r(np.transpose(luwe),np.transpose(data))
# print(output.summary())

std_err = output.bse
robust_err = output_robust.bse
coeff = output.params
print('\n''The normal linear regression model for log weekly wages is defined as: ')
print('luwe = %0.4f + %0.4f*educ + %0.4f*experience + %0.4f*experience^2' % (coeff[3], coeff[2], coeff[1], coeff[0]))

print('\n              Beta0  Beta1  Beta2  Beta3')
print('Std Error:    %0.4f %0.4f %0.4f %0.4f' % (std_err[3], std_err[2], std_err[1], std_err[0]))
print('Robust Error: %0.4f %0.4f %0.4f %0.4f' % (robust_err[3], robust_err[2], robust_err[1], robust_err[0]))

# Variance / Covariance Matrix
print('\n')
var_cov_matrix = output.cov_params()
residual_var = np.cov(output.resid)

var_cov_matrix_df = pd.DataFrame(data=var_cov_matrix, 
                             index=['educ', 'exper', 'exper^2', 'K'],
                             columns=['educ', 'exper', 'exper^2', 'K'])

print('Variance / Covariance Matrix\n')
print(var_cov_matrix_df.round(5)) 
print('\n''Residual Variance: %0.4f' % residual_var)
print('\n')

#--------------------------------------------------------
# Problem 3 - Using sample 1 as base case
print('-'*40)
print('\n''Problem 3')
educ_increase = -1
educ_new = educ + educ_increase
exper_new = exper(age,educ_new)

luwe_old = coeff[3] + coeff[2]*educ + coeff[1]*experience + coeff[0]*np.square(experience)
luwe_new = coeff[3] + coeff[2]*educ_new + coeff[1]*exper_new + coeff[0]*exper_new**2
luwe_delta = np.mean(luwe_new - luwe_old) # Delta method of average level of earnings
print('\n''A one year reduction in education level results in a change\n in log earnings of %0.4f.' % luwe_delta)
print('\n')

#--------------------------------------------------------
# Problem 4
print('-'*40)
print('\n''Problem 4''\n')
answer1 = 'Yes. We can obtain the same effect with redefined covariates using algebra.\n'
answer2 = 'Define luwe delta as theta.\nTheta = -beta1 + beta2 + beta3(2*experience -1)'
answer3 = 'Rearrange and solve for beta1'
answer4 = 'beta1 = beta2 + beta3*(2*experience + 1) - theta'
answer5 = 'Insert beta1 equation into linear model'
answer6 = '\nlog earnings = beta0 - theta*educ + beta2*(educ + experience) + beta3*(experience^2 + (2*experience +1)*educ) + residual error'
print(answer1,answer2,answer3,answer4,answer5,answer6, sep='\n')
print('\n')

#--------------------------------------------------------
# Problem 5
print('-'*40)
print('\n''Problem 5''\n')

i = 0
educ_min_12 = educ.copy()
for item in educ:
    educ_min_12[i] = max(12,item)
    i += 1
  
experience_new = exper(age,educ_min_12)
exp_sq_new = np.square(experience_new)

theta0 = np.exp(coeff[3] + coeff[2]*educ + coeff[1]*experience + coeff[0] * exp_sq + residual_var/2)
theta1 = np.exp(coeff[3] + coeff[2]*educ_min_12 + coeff[1]*experience_new + coeff[0] * exp_sq_new + residual_var/2)
theta_hat = theta1 - theta0 # impact of earnings from new policy for each sample
theta_hat_mean = np.mean(theta_hat) # average earnings impact from new policy

print('The policy of minimizing all education to 12 years results in a change of\n%0.4f in the average level of earnings.' % (theta_hat_mean))
print('\n')

# Problem 6
print('-'*40)
print('\n' 'Problem 6' '\n')
# 'np' stands for 'new policy'
luwe_np = np.log(np.exp(luwe) + theta_hat) # new earnings estimate with new policy
std_err_np = np.var(luwe_np)**.5 # variance of new earnings aka standard error of policy
print('For the conventional standard errors, the standard error of the new policy is calculated to be %0.4f.' % std_err_np)

print('\n')
coeff = output_robust.params
theta0 = np.exp(coeff[3] + coeff[2]*educ + coeff[1]*experience + coeff[0] * exp_sq + residual_var/2)
theta1 = np.exp(coeff[3] + coeff[2]*educ_min_12 + coeff[1]*experience_new + coeff[0] * exp_sq_new + residual_var/2)
theta_hat = theta1 - theta0
theta_hat_mean = np.mean(theta_hat)
luwe_np = np.log(np.exp(luwe) + theta_hat)
std_err_np = np.var(luwe_np)**.5

print('For the robust standard errors, the standard error of the new policy is calculated to be %0.4f, effectively the same as conventional.' % std_err_np)
print('\n')

# Problem 7
print('-'*40)
print('\n' 'Problem 7' '\n')
from numpy import random

r = 1000 # iterations of the data

data = np.array(data)
luwe = np.array(luwe)
matrix = np.c_[luwe, data]

beta_bootstrap = np.zeros((r,k))
matrix_random = np.zeros((n,k))
theta_hat_mean = []

for i in range(0,r):
    for k in range(0,n):
        random_row = random.randint(n) # Random row to select from data
        matrix_random[k] = matrix[random_row] # random row selected in new sample set
    
    luwe_r = matrix_random[:,0] # log earnings
    data_r = matrix_random[:,1:4] # Education, experience, and experience^2
            
    output_np_bstrap = reg_m(np.transpose(luwe_r),np.transpose(data_r)) # OLS function
    
    beta_bstrap = output_np_bstrap.params # Coefficients
    r_var_bstrap = np.var(output_np_bstrap.resid) # Residual variance
    
    i = 0
    educ_min_12 = educ.copy()
    for item in educ:
        educ_min_12[i] = max(12,item) # Education with a minimum of 12
        i += 1
    
    experience_new = exper(age,educ_min_12) # Recalculating experience with education at a minimum of 12
    exp_sq_new = np.square(experience_new)
    
    theta0 = np.exp(beta_bstrap[3] + beta_bstrap[2]*educ + beta_bstrap[1]*experience + beta_bstrap[0] * exp_sq + r_var_bstrap/2)
    theta1 = np.exp(beta_bstrap[3] + beta_bstrap[2]*educ_min_12 + beta_bstrap[1]*experience_new + beta_bstrap[0] * exp_sq_new + r_var_bstrap/2)
    theta_hat = theta1 - theta0 # New policy less original
    theta_hat_mean.append(np.mean(theta_hat)) # Theta mean
    
std_err_bstrap = np.var(theta_hat_mean, ddof = 1)**0.5 # std dev of theta mean aka policy error
print('For the nonparametric bootstrap method, the standard error of the new policy is calculated to be %0.4f.' % std_err_bstrap)

print('\n''The bootstrap standard error is marginally lower than the analytical standard errors for both conventional and robust errors.')

