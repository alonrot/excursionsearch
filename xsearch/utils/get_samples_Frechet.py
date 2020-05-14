# Copyright 2020 Max Planck Society. All rights reserved.
# 
# Author: Alonso Marco Valle (amarcovalle/alonrot) amarco(at)tuebingen.mpg.de
# Affiliation: Max Planck Institute for Intelligent Systems, Autonomous Motion
# Department / Intelligent Control Systems
# 
# This file is part of excursionsearch.
# 
# excursionsearch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
# 
# excursionsearch is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# excursionsearch.  If not, see <http://www.gnu.org/licenses/>.
#
#
import numpy as np
import torch
from scipy.special import erfc
import matplotlib.pyplot as plt
import pdb
import logging
logger = logging.getLogger(__name__)

def get_fmin_samples_from_gp(model,Nsamples=10,gridSize=1000,eta=None):
    '''
    Call a routine that samples from the Frechet distribution that models
    the minimum of the objective. See Sec. 3.1 and 3.2 in the paper.
    '''
    assert eta is not None, "This function assumes that super().update_eta() has been called and returned succesfully"
        
    # Correct eta_c value to avoid numerical unstability:
    correctness = 3.0*torch.sqrt(model.likelihood.noise[0]) # noise (fixed)
    eta -= correctness

    # Collect the samples following a similar a approach to (Wang & Jegelka, 2017, Appendix B):
    f_min_samples = sample_fmin_Frechet(model,lb=0.0,hb=1.0,nMs=Nsamples,gridSize=gridSize,eta_c=eta)
    return torch.from_numpy(f_min_samples.copy()).to(device="cpu",dtype=torch.float32)

def sample_fmin_Frechet(model,lb,hb,nMs=10,gridSize=1000,eta_c=None,plot_PDF=False,
                        model_cons=None,delta_cons=None,return_also_quantiles=False,return_parameters=False):
    """
    Authorship: This function is substantially based on an existing python implementation by other authors.
    To keep the authorship confidentiality, we omit the names of the original authors.
    In the final (public) version of the code, we will put them back and acknowledge them.

    Notes: This function was originally created from the perspective of finding samples of the maximum
    So, we flip the GP mean and the observations, and compute their maximums. Then we flip the sign back.
    """

    # Obtain observations:
    x_ob = model.train_inputs[0].detach().cpu().numpy() # Assume x is [N x dim]
    # x_ob = np.copy(model.get_X()) # Assume x is [N x dim]
    # y_ob = - np.copy(model.get_Y()) # Minus for minimization
    dim = model.dim
    eta_c = eta_c.squeeze().item()
    
    # Obtain obsrvation noise:
    noise_var = model.likelihood.noise[0].item()
    # pdb.set_trace()

    # Remove those points where the constraint is violated:
    if model_cons is not None and delta_cons is not None:

        Ntrials = 10
        ncount = 0
        safe_area_found = False
        while safe_area_found == False and ncount < Ntrials:
    
            # Get lower bound and upper bound (lb and hb), [dim x 1]
            Xgrid = np.tile(lb, (gridSize,1)) + np.tile((hb-lb), (gridSize,1)) * np.random.rand(gridSize,dim)
            Xgrid = np.vstack((Xgrid, x_ob))

            ind_cons_sat = model_cons.get_cdf_untill_thres_vec(Xgrid,0.0) > delta_cons
            if np.any(ind_cons_sat): # Do this ONLY if the constraint satisfied somewhere
                safe_area_found = True
                Xgrid = np.atleast_2d(Xgrid[ind_cons_sat,:])

            ncount = ncount + 1
    else:
        Xgrid = np.tile(lb, (gridSize,1)) + np.tile((hb-lb), (gridSize,1)) * np.random.rand(gridSize,dim)


    sx = Xgrid.shape[0]

    # Prediction at grid locations:
    # muVector, cov_mat = model.prediction(Xgrid)
    prediction = model(torch.from_numpy(Xgrid).to(device="cpu",dtype=torch.float32))
    muVector = prediction.mean.detach().cpu().numpy()
    cov_mat = prediction.covariance_matrix.detach().cpu().numpy()
    muVector = np.atleast_2d(muVector).T
    varVector = np.diag(cov_mat)
    varVector = varVector.clip(noise_var)
    stdVector = np.atleast_2d(np.sqrt(varVector)).T

    # Flip sign to find samples of fmin, instead of fmax:
    muVector = - muVector 

    def probf(m0):
        z = (m0 - muVector)/stdVector
        cdf = 0.5 * erfc(-z / np.sqrt(2))
        return np.prod(cdf)

    # left = np.max(y_ob)
    left = -eta_c # Minus for minimization
    f_max_samples = np.zeros(nMs)
    Nrep = 100
    if probf(left) < 0.25:
        right = np.max(muVector + 5 *stdVector)
        while probf(right) < 0.75:
            right = right + right - left

        mgrid = np.linspace(left,right,Nrep)
        # mgrid = 1xNrep , muVector = sx x 1, stdVector = sx x 1
        z_grid = (np.tile(mgrid,(sx,1)) - np.tile(muVector,(1,Nrep))) / np.tile(stdVector,(1,Nrep))
        z_cdf = 0.5 * erfc(-z_grid / np.sqrt(2))
        prob = np.prod(z_cdf, axis=0) [None,:]
        
        # Find quantiles
        q1 = find_between(0.25, probf, prob, mgrid, 0.01)
        q2 = find_between(0.75, probf, prob, mgrid, 0.01)

        # Approximate the Frechet parameters alpha and beta.
        eta_c_flip = -eta_c

        c0 = np.log(4.)/np.log(4./3.)
        alpha_exponent = np.log((q1-eta_c_flip)/(q2-eta_c_flip))/np.log(c0) # This is not alpha, but -1/alpha
        s = (q1-eta_c_flip) / ((np.log(4.))**alpha_exponent) # Double check this line, as it is diferent in the pdf

        epsi = 0.05
        check_alpha = alpha_exponent >= -(1.0+epsi) and  alpha_exponent < 0.0# I.e., alpha > 1
        check_std =  s > 0.0

        if check_alpha == False or check_std == False:
            maxes_samples = q2
            logger.info("@sample_fmin_Frechet: alpha < 1.0")
            logger.info("Return all samples at the second quantile")
        else:
            maxes_samples = eta_c_flip + s*(-np.log(np.random.rand(1,nMs)))**alpha_exponent

        if plot_PDF == True:
            Ndiv = 101
            f_vec = np.linspace(eta_c_flip , eta_c_flip + 3.0 , Ndiv)
            CDF_Frechet = np.exp(-((f_vec-eta_c_flip)/s)**(1./alpha_exponent))
            
            alpha = -1./alpha_exponent
            PDF_Frechet = CDF_Frechet * (alpha/s)*((f_vec-eta_c_flip)/s)**(-1.-alpha)

            hdl_fig,hdl_plot = plt.subplots(2,1)
            hdl_plot[0].plot(f_vec,CDF_Frechet)
            hdl_plot[1].plot(f_vec,PDF_Frechet)
            plt.show(block=False)
            plt.pause(5)

        # maxes_samples[np.where( maxes_samples < 0.5)] = left + 5.0 * np.sqrt(noise_var) # new: Commented this out, as it seems strange...
        f_max_samples[:] = maxes_samples

    else:
        # fac = 5.0 # original
        fac = 10.0 # new
        logger.info("Return all samples at -(-eta_c + {0:f}*std_n)".format(fac))
        f_max_samples[:] = left + fac * np.sqrt(noise_var) # original
        # f_max_samples[:] = left + 10.0 * np.sqrt(noise_var) # new
        # f_max_samples[:] = q2 # new

    if f_max_samples is None or np.any(np.isnan(f_max_samples)) or np.any(np.isinf(f_max_samples)):
        logger.info("f_max_samples not well defined...")
        pdb.set_trace() # TODO: Replace by raise

    # Flip and sort:
    f_max_samples = -f_max_samples # Flip sign, as we originally wanted to find minimums
    f_max_samples = np.sort(f_max_samples) # Ascending order by default
    f_max_samples = f_max_samples[::-1] # Reverse order

    if return_also_quantiles == True and probf(left) < 0.25:
        q25 = -q1
        q75 = -q2
        median_flipped = eta_c_flip + s*(np.log(2.))**(alpha_exponent) # Note: 'alpha_exponent' is not alpha, but -1/alpha
        median = -median_flipped
        return f_max_samples,q75,median,q25
    elif return_also_quantiles == True:
        return f_max_samples,f_max_samples[0],f_max_samples[0],f_max_samples[0]
    elif return_parameters == True and probf(left) < 0.25:
        return f_max_samples,alpha_exponent,s
    else:
        return f_max_samples


def find_between(val, func, funcvals, mgrid, thres):
    """
    Authorship: This function is substantially based on an existing python implementation by other authors.
    To keep the authorship confidentiality, we omit the names of the original authors.
    In the final (public) version of the code, we will put them back and acknowledge them.

    """
    t2 = np.argmin(abs(funcvals - val))
    check_diff = abs(funcvals[0,t2] - val)
    if abs(funcvals[0,t2] - val) < thres:
        res = mgrid[t2]
        return res

    assert funcvals[0,0] < val and funcvals[0,-1] > val
    if funcvals[0,t2] > val:
        left = mgrid[t2 - 1]
        right = mgrid[t2]
    else:
        left = mgrid[t2]
        right = mgrid[t2 + 1]

    mid = (left + right) / 2.0
    midval = func(mid)
    cnt = 1
    while abs(midval - val) > thres:
        if midval > val:
            right = mid
        else:
            left = mid

        mid = (left + right) / 2.
        midval = func(mid)
        cnt = cnt + 1
        if cnt > 10000:
            pdb.set_trace()
    res = mid
    return res