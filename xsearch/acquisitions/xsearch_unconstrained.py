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
import torch
from torch import Tensor
from typing import Optional
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
import numpy as np # These two libraries need to dissapear, as allt he code should be in torch
from scipy.stats import norm # These two libraries need to dissapear, as allt he code should be in torch
from scipy.special import erf
from botorch.optim import optimize_acqf
from .acquisition_base import AcquisitionBaseTools
from xsearch.utils.get_samples_Frechet import get_fmin_samples_from_gp
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.gen import gen_candidates_scipy
from botorch.gen import get_best_candidates
import pdb

# class Xsearch(MCAcquisitionFunction):
class Xsearch(AnalyticAcquisitionFunction,AcquisitionBaseTools):
	def __init__(self, model: Model, options: dict) -> None:
		# MCAcquisitionFunction.__init__(self, model=model, sampler=sampler, objective=IdentityMCObjective())
		AnalyticAcquisitionFunction.__init__(self, model=model, objective=ScalarizedObjective(weights=torch.Tensor([1.0])))
		AcquisitionBaseTools.__init__(self,model=model, iden="Xsearch", Nrestarts_eta=options["Nrestarts_eta"])

		self.u_vec = None
		self.Nsamples_fmin = options["Nsamples_fmin"]
		self.Nrestarts = options["Nrestarts_safe"]
		self.debug = False
		self.method = options["method_safe"]
		self.disp_info_scipy_opti = options["disp_info_scipy_opti"]
		self.dim = self.model.dim

	@t_batch_mode_transform(expected_q=1)
	def forward(self, X: Tensor) -> Tensor:
		"""Evaluate scalarized qUCB on the candidate set `X`.

		Args:
		    X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
		        design points each.

		Returns:
		    Tensor: A `(b)`-dim Tensor at the given design points `X`.
		"""

		assert self.u_vec is not None, "self.u_vec must be updated before calling forward(). Use self.get_next_point() to optimize Xsearch"

		if self.debug:
			pdb.set_trace()
		
		# Get indices:
		Npred = X.shape[0]
		dim = X.shape[-1]
		Nfcmin_samples = self.u_vec.shape[0]

		# Error checking:
		assert X.dim() == 3
		assert self.u_vec is not None, "Call self.update_u_vec() first"

		dmpred_dx_cond,\
		stdpred_dx_cond = self.model.predict_gradient_at_test_location_conditioned_on_u(test_x_vec=X, # [b x (q=1) x dim]
																																										u_vec=self.u_vec,
																																										sequential=True)
		# Sanity check:
		assert dmpred_dx_cond.shape[0] == Npred
		assert dmpred_dx_cond.shape[1] == dim
		assert Nfcmin_samples == dmpred_dx_cond.shape[2] == stdpred_dx_cond.shape[2]
		
		# Get posterior GP conditioned on data, to obtain p(u|D)
		posterior_X = self.model(X)
		# pdf_u_cond_vec = torch.exp(posterior_X.log_prob(self.u_vec.unsqueeze(0).unsqueeze(2)))
		pdf_u_cond_vec = torch.exp(posterior_X.log_prob(self.u_vec.unsqueeze(1).unsqueeze(1)))
		pdf_u_cond_vec = pdf_u_cond_vec.T
		assert pdf_u_cond_vec.shape == torch.Size([Npred,Nfcmin_samples]) # [Npred x Nfcmin_samples]


		val =	self.alpha_xsearch(dmpred_dx_cond,stdpred_dx_cond,pdf_u_cond_vec) # 1D vector
		# print("@alpha_xsearch: X: ",X)
		# print("@alpha_xsearch: X.shape: ",X.shape)
		# print("@alpha_xsearch: val.shape: ",val.shape)


		return val

	@staticmethod
	def get_crossings_functional(mufp,stdfp):
		'''
		Corresponds to eq. (7) in the paper
		'''
		val_vec = 2*stdfp*norm.pdf(mufp/stdfp) + mufp*erf(mufp/(stdfp*np.sqrt(2))) # numpy
		return torch.from_numpy(val_vec).to(device="cpu",dtype=torch.float32)

	@staticmethod
	def alpha_xsearch(dmpred_dx_cond,stdpred_dx_cond,pdf_u_cond_vec):
		'''
		Corresponds to eq. (10) in the paper
		'''

		# Get the rest of the parts of the acquisition function:
		CROSS_vec_per_dim = torch.zeros((dmpred_dx_cond.shape))
		dim = dmpred_dx_cond.shape[1]
		Nfcmin_samples = dmpred_dx_cond.shape[2]
		for ii in range(Nfcmin_samples):
			
			# Crossings contribution:
			for jj in range(dim):
				CROSS_vec_per_dim[:,jj,ii] = Xsearch.get_crossings_functional(dmpred_dx_cond[:,jj,ii].detach().cpu().numpy(),\
																																	stdpred_dx_cond[:,jj,ii].detach().cpu().numpy())

		# Assume L-1 norm:
		CROSS_vec = torch.sum(CROSS_vec_per_dim,axis=1)

		# Put together both contributions:
		CROSS_val_all_fmin_samples = pdf_u_cond_vec*CROSS_vec

		# Compute the mean on the fmin_samples:
		CROSS_val = torch.mean(CROSS_val_all_fmin_samples,axis=1)

		return CROSS_val

	def update_u_vec(self,u_vec):
		"""
		u_vec represents is a tensor that contains the samples of the minimum.
		This function is not strictily necessary, as self.u_vec can be set directly, 
		but left for resolution.
		"""
		self.u_vec = u_vec

	def get_next_point(self):

		# Find and store the minimum of the posterior mean, i.e., min_x mu(x|D), where D is the data set D={Y,X}, and mu(x|D)
		# is the posterior mean of the GP queried at location x
		super().update_eta()

		# Gather fmin samples, using the Frechet distribution:
		fmin_samples = get_fmin_samples_from_gp(model=self.model,Nsamples=self.Nsamples_fmin,eta=self.eta) # This assumes self.eta has been updated
		self.update_u_vec(fmin_samples)

		self.my_print("[get_next_point()] Computing next candidate by maximizing the acquisition function ...")
		options={"batch_limit": 50,"maxiter": 200,"ftol":1e-9,"method":self.method,"iprint":2,"maxls":20,"disp":self.disp_info_scipy_opti}

		if self.dim > 2:
			self.x_next, self.alpha_next = self.optimize_acqui_use_restarts_as_batch(options)
		else:
			self.x_next, self.alpha_next = self.optimize_acqui_use_restarts_individually(options)
		
		self.my_print("Done!")
		self.my_print("xnext: " + str(self.x_next.view((1,self.dim)).detach().cpu().numpy()))
		self.my_print("alpha_next: {0:2.2f}".format(self.alpha_next.item()))

		return self.x_next,self.alpha_next
