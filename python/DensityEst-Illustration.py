# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### Parametric Estimation of Bin-based Density Forecast in Surveys 
#
# - Following [Manski et al.(2009)](https://www.tandfonline.com/doi/abs/10.1198/jbes.2009.0003)
# - Three cases 
#    - case 1. 3+ intervals with positive probabilities, or 2 intervals with positive probabilities but open-ended from either end, to be fitted with a generalized beta distribution
#    - case 2. exactly 2 adjacent and close-ended bins positive probabilities, to be fitted with a triangle distribution 
#    - case 3. __one or multiple__ adjacent intervals with equal probabilities, to be fitted with a uniform distribution
#    - cases excluded for now:
#      - nonadjacent bins with positive probabilities with bins with zero probs in between 
#      -  only one bin with positive probabilities at either end 
#    

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# +
## import all functions from DensityEst 
## see DensityEst.ipynb for detailed explanations 

from DensityEst import GeneralizedBetaEst,GeneralizedBetaStats,TriangleEst,TriangleStats,UniformEst,UniformStats,SynDensityStat
# -

# ### Test using made-up data

# + code_folding=[]
## test 1: GenBeta Dist
sim_bins= np.array([0,0.2,0.32,0.5,1.6,2])
sim_probs=np.array([0.0,0.2,0.3,0.3,0.2])
para_est = GeneralizedBetaEst(sim_bins,sim_probs)
print(para_est)
if len(para_est) != 0:
    GeneralizedBetaStats(para_est[0],
                        para_est[1],
                        para_est[2],
                        para_est[3])
else:
    print('no estimation')

# + code_folding=[]
## test 2: Triangle Dist
sim_bins2 = np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs2=np.array([0,0.2,0.8,0,0])
para_est = TriangleEst(sim_bins2,sim_probs2)
print(para_est)
TriangleStats(para_est['lb'],
              para_est['ub'], 
              para_est['mode'])

# + code_folding=[]
## test 3: Uniform Dist with one interval
sim_bins3 = np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs3 = np.array([0,0,1,0,0])
para_est= UniformEst(sim_bins3,sim_probs3)
print(para_est)
UniformStats(para_est['lb'],para_est['ub']) 

# + code_folding=[]
## test 4: Uniform Dist with multiple adjacent bins with same probabilities 
sim_bins4 = np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs4=np.array([0.2,0.2,0.2,0.2,0.2])
para_est= UniformEst(sim_bins4,
                     sim_probs4)
print(para_est)
UniformStats(para_est['lb'],
             para_est['ub']) 


# + code_folding=[]
## test 5: Uniform Dist with multiple non-adjacent bins with equal probabilities
## totally excluded from current estimation 

sim_bins5 = np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs5= np.array([0,0.5,0,0.5,0])
para_est = UniformEst(sim_bins5,
                     sim_probs5)
print(para_est)
UniformStats(para_est['lb'],
             para_est['ub']) 
# -

# ### Test with simulated data from known distribution 
# - we simulate data from a true beta distribution with known parameters
# - then we estimate the parameters with our module and see how close it is with the true parameters 

# + code_folding=[]
## simulate a generalized distribution
sim_n=1000
true_alpha,true_beta,true_loc,true_scale=0.5,0.7,0.0,1.0

sim_data = beta.rvs(true_alpha,
                    true_beta,
                    loc= true_loc,
                    scale=true_scale,
                    size=sim_n)

sim_bins2=plt.hist(sim_data)[1]
sim_probs2=plt.hist(sim_data)[0]/sim_n

sim_est = GeneralizedBetaEst(sim_bins2,
                           sim_probs2)
print('Estimated parameters',sim_est)


# +
print('Estimated moments:',GeneralizedBetaStats(sim_est[0],
                          sim_est[1],
                          sim_est[2],
                          sim_est[3]))

print('True simulated moments:',
      np.mean(sim_data),
     np.std(sim_data)**2,
     np.std(sim_data)
     )
# -

## plot the estimated generalized beta versus the histogram of simulated data drawn from a true beta distribution 
sim_x = np.linspace(true_loc,true_loc+true_scale,sim_n)
sim_pdf=beta.pdf(sim_x,sim_est[0],sim_est[1],loc=true_loc,scale=true_scale)
plt.plot(sim_x,sim_pdf,label='Estimated pdf')
plt.hist(sim_data,density=True,label='Dist of Simulated Data')
plt.legend(loc=0)

# + code_folding=[]
## testing the synthesized estimator function using an arbitrary example created above
print(SynDensityStat(sim_bins2,sim_probs2)['variance'])
print(SynDensityStat(sim_bins2,sim_probs2)['iqr1090'])
# -

# ### Estimation with real sample data (from SPF)

# + code_folding=[]
### loading probabilistic data  
sample_data =pd.read_stata('../data/sample_data.dta')   
## this is the quarterly SPF forecast on 1-year ahead core inflation 
# -

sample_data.tail()

# + code_folding=[]
## survey-specific parameters 
nobs=len(sample_data)
sample_bins=np.array([-10,0,0.5,1,1.5,2,2.5,3,3.5,4,10])
print("There are "+str(len(sample_bins)-1)+" bins in the sample data")

# + code_folding=[0]
##############################################
### attention: the estimation happens here!!!!!
###################################################


## creating positions 
index  = sample_data.index

columns=['Mean',
         'Var', 
         'Std',
         'Iqr1090']

sample_moment_est = pd.DataFrame(index=index,
                                 columns=columns)

## Invoking the estimation

for i in range(nobs):
    print(i)
    ## take the probabilities (flip to the right order, normalized to 0-1)
    PRCCPI_y0 = np.flip(np.array([sample_data.iloc[i,:]['PRCCPI'+str(n)]/100 for n in range(1,11)]))
    print(PRCCPI_y0)
    if not np.isnan(PRCCPI_y0).any():
        stats_est=SynDensityStat(sample_bins,PRCCPI_y0)
        if stats_est is not None and len(stats_est)>0:
            sample_moment_est['Mean'][i]= stats_est['mean']
            print(stats_est['mean'])
            sample_moment_est['Var'][i]=stats_est['variance']
            print(stats_est['variance'])
            sample_moment_est['Std'][i]=stats_est['std']
            print(stats_est['std'])
            sample_moment_est['Iqr1090'][i]=stats_est['iqr1090']
            print(stats_est['iqr1090'])           
# -

### exporting moments estimates to pkl
data_est = pd.concat([sample_data,sample_moment_est], join='inner', axis=1)
data_est.to_pickle("./DstSampleEst.pkl")
data_pkl = pd.read_pickle('./DstSampleEst.pkl')

# +
## convert all moments to numeric values 

data_pkl['Mean']=pd.to_numeric(data_pkl['Mean'],errors='coerce')   # CPI from y-1 to y
data_pkl['Var']=pd.to_numeric(data_pkl['Var'],errors='coerce')
data_pkl['Std']=pd.to_numeric(data_pkl['Std'],errors='coerce')
data_pkl['Iqr1090']=pd.to_numeric(data_pkl['Iqr1090'],errors='coerce')
# -


data_pkl.tail()

## export estimates 
data_pkl.to_excel('../data/Dstsample_data.xlsx')

# +
### Robustness checks: focus on NaN estimates (due to missing data or non-convergence)

sim_bins_data = sample_bins
nan_est = data_pkl['Mean'].isna()
missing_data = data_pkl['PRCCPI1'].isna() ## no data to estimate in the first place 

print(str(sum(nan_est))+' missing estimates')
print(str(sum(missing_data))+' of which is due to missing data')

print('\n')
print('All nan estimates due to other reasons\n')
ct=0
figure=plt.plot()
for id in data_pkl.index[(nan_est) & (~missing_data)]:
    print(id)
    print(data_pkl['Mean'][id])
    sim_probs_data= np.flip(np.array([sample_data['PRCCPI'+str(n)][id]/100 for n in range(1,11)]))
    plt.bar(sim_bins_data[1:],sim_probs_data)
    print(sim_probs_data)
    ## do estimation again 
    stats_est=SynDensityStat(sample_bins,
                             sim_probs_data)
    if stats_est is not None:
        print(stats_est['mean'])
    else:
        print('Estimate is None!!!')
# -

