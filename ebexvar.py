import sys, os
import numpy as np
from astropy.table import Table
from cmdstanpy import CmdStanModel 
from scipy.stats import poisson

def estimate_meanCR(data):

    NC = data['NC']
    NS = data['NS']    
    NI = data['NI']
    
    LGCRMEAN=[]
    for i in range(NS):

        i1 = data['indices'][i]
        i2=  data['indices'][i+1]
        c = data['counts'][i1:i2]
        b = data['bkg'][i1:i2]
        e = data['time'][i1:i2]

        LGCRMEAN.append(getPDF(e.sum(), b.sum(), c.sum()))

    return  np.array(LGCRMEAN)

def getPDF(exp, bkg, total):

    nmin = getSenseLimit(pfalse=4e-6,bkg=bkg)

    lgf_min = np.log10( ( nmin - bkg ) / exp ) - 2.0
    if(total>nmin):
        lgf_max = np.log10( ( total - bkg ) / exp ) + 1.0
    else:
        lgf_max = np.log10( ( nmin - bkg ) / exp ) + 1.0


    nmax_lgf=1000
    lgflux=np.linspace(lgf_min,lgf_max, nmax_lgf, endpoint=True, retstep=False)
    flux=10**lgflux
    expected  = flux * exp + bkg
    p = poisson(expected)
    prob=p.pmf(total) 
    
    imode = np.argmax(prob)
    
    return lgflux[imode]


def getSenseLimit(pfalse=4e-6,bkg=10):

    min_counts_detect = poisson.isf(pfalse, bkg)
    
    return min_counts_detect;


def create_dict(data):

    dict={'NS': 0, 'NI': 0,  'NC':0, 'indices': [], 'SRCID':[], 'counts': [], 'time': [], 'bkg': [], 'DTYR':[]}

    unique, unique_indices, unique_counts = np.unique(data['SRCID'].data, return_index=True, return_inverse=False, return_counts=True)

    dict['indices'] = unique_indices
    dict['indices'] =  np.append(dict['indices'], [len(data)]) # adding the last element
    
    dict['NI'] =  len(dict['indices'])
    dict['NS'] = len(unique)

    dict['NC']     =  len(data['counts'].data)    
    dict['counts'] = data['counts'].data
    dict['time']   = data['time'].data
    dict['bkg']    = data['bkg'].data
    dict['DTYR']   = data['DTYEARS'].data

    return dict; 

def init_function(seed):

    # guess good parameters for chain to start
    rng = np.random.RandomState(seed)

    guess = dict(
        # start with short time-scales: all bins independent
        LGCR_MEAN=estimate_meanCR(data_dict),
        # A 
        A = np.log10(0.1),
        # B
        B = 0.5,
        # helper parameter 
        raw_sigma=rng.normal(size=data_dict['NS']),
        # helper parameter 
        raw=rng.normal(size=data_dict['NC']),
    )

    return guess


def get_params(fit):

    var = fit.stan_variables()
    res = {}

    for v in var.keys():
        shape = var[v].shape
        print(v, shape)
        if(len(shape)==1):
            res["{}_q".format(v)]=np.quantile(var[v], q=[0.5, 0.16, 0.84, 0.05, 0.95])
        res["{}_c".format(v)] = var[v]
        
    return res
        

def main():

    global data_dict

    seed = 1
    
    # compile the stan model
    model = CmdStanModel(stan_file="STANMODELS/eBExVar.stan", cpp_options={'STAN_THREADS':'true'})#, force_compile=True)

    # read the file with the light curves 
    data = Table.read("DATA/mock_lcs.fits")

    # organise the data in a data dictionary that will be read by the stan code
    data_dict = create_dict(data)

    # add an estimate of the mean count rate of each source
    data_dict['lgf0']=estimate_meanCR(data_dict)

    # run stan
    fit = model.sample(data=data_dict, chains=4,
                       iter_warmup=3000,
                       iter_sampling=2000,
                       inits=init_function(seed),
                       seed=seed,
                       adapt_delta=0.95,
                       max_treedepth=12,                       
                       show_console=True,
                       show_progress=True)

    # print fit diagnostics
    print(fit.diagnose())

    # print fit summary
    print(fit.summary())

    # export fit results to a dictionary
    res = get_params(fit)

    # print out the quantiles of the NEV estimat
    print("NEV quantiles:", np.quantile(10**res['A_c'],q=[0.5, 0.16, 0.84]))

if __name__=="__main__":
    main()

