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
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.means import ConstantMean, ConstantMeanGrad, ZeroMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, RBFKernelGrad
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
import random
import numpy as np
import torch
from torch import Tensor
import pdb
from xsearch.utils.plotting_collection import PlotProbability
from xsearch.utils.parsing import extract_prior
from gpytorch.priors import GammaPrior
from .gpmodel_grad import GPmodelWithGrad
import matplotlib.pyplot as plt
from typing import Optional, Tuple
np.set_printoptions(linewidth=10000)

class GPmodel(ExactGP, GPyTorchModel):
	"""

	This model internally instantiates another "virtual" model to compute the derivatives.
	Unfortunately, we can't have just a single model that does everything, because botorch/gpytorch
	does not support using this model without training data that includes derivatie observations.
	In our case, derivative observations are not mandatory, and when not present, 
	such model can't be used.

	Use alternatively a BOTorch model:
	https://botorch.org/api/models.html?highlight=singletaskgp#botorch.models.gp_regression.FixedNoiseGP
	This model includes the next features:
	https://botorch.org/api/models.html?highlight=singletaskgp#botorch.models.model.Model
	"""
	_num_outputs = 1  # to inform GPyTorchModel API

	def __init__(self, train_X: Tensor, train_Y: Tensor, options: dict, which_type: Optional[str] = "obj") -> None:

		# Error checking:
		assert train_Y.dim() == 1, "train_Y is required to be 1D"
		self._validate_tensor_args(X=train_X, Y=train_Y[:,None]) # Only for this function, train_Y must be 2D (this must be a bug in botorch)

		# Dimensionality of the input space:
		self.dim = train_X.shape[-1]

		# Model identity:
		self.iden = "GP_model_{0:s}".format(which_type)

		# Likelihood:
		noise_std = options["noise_std_obj"]
		lik = FixedNoiseGaussianLikelihood(noise=torch.full_like(train_Y, noise_std**2))

		# Initialize parent class:
		super().__init__(train_X, train_Y, lik)

		# Obtain hyperprior for lengthscale and outputscale:
		# NOTE: The mean (zero) and the model noise are fixed
		lengthscale_prior, outputscale_prior = extract_prior(options,which_type)

		# Initialize prior mean:
		# self.mean_module = ConstantMean()
		self.mean_module = ZeroMean()

		# Initialize covariance function:
		# base_kernel = RBFKernel(ard_num_dims=train_X.shape[-1],lengthscale_prior=GammaPrior(3.0, 6.0)) # original
		# self.covar_module = ScaleKernel(base_kernel=base_kernel,outputscale_prior=GammaPrior(2.0, 0.15)) # original
		base_kernel = RBFKernel(ard_num_dims=self.dim,lengthscale_prior=lengthscale_prior,lengthscale_constraint=GreaterThan(1e-2))
		self.covar_module = ScaleKernel(base_kernel=base_kernel,outputscale_prior=outputscale_prior)

		# Make sure we're on the right device/dtype
		self.to(train_X)

		# Instantiate the gradient model:
		self.model_grad = GPmodelWithGrad(dim=self.dim)

	def set_hyperparameters(self,lengthscale,outputscale,noise):
		self.covar_module.base_kernel.lengthscale = lengthscale
		self.covar_module.outputscale = outputscale
		self.likelihood.noise[:] = noise
		# self.mean_module.constant[:] = 0.0 # Assume zero mean

	def display_hyperparameters(self):
		self.my_print("  Re-optimized hyperparameters")
		self.my_print("  ----------------------------")
		self.my_print("    Outputscale (stddev) | {0:2.4f}".format(self.covar_module.outputscale.item()))
		self.my_print("    Lengthscale(s)       | " + str(self.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()))

	def update_hyperparameters(self):
		print("\n")
		self.my_print("Fitting model...")
		self.my_print("----------------")
		mll = ExactMarginalLogLikelihood(self.likelihood, self)
		fit_gpytorch_model(mll,max_retries=10) # See https://botorch.org/api/_modules/botorch/optim/utils.html#sample_all_priors
														# max_retries: https://botorch.org/api/_modules/botorch/fit.html#fit_gpytorch_model

		# Sometimes fit_gpytorch_model fails:
		if torch.any(self.covar_module.base_kernel.lengthscale > 2.0):
			self.covar_module.base_kernel.lengthscale[:] = 2.0
		if torch.any(self.covar_module.outputscale > 2.0):
			self.covar_module.outputscale[:] = 2.0

		self.update_hyperparameters_of_model_grad()

	def update_hyperparameters_of_model_grad(self):
		self.model_grad.update_hyperparameters(	lengthscale=self.covar_module.base_kernel.lengthscale,
																						outputscale=self.covar_module.outputscale,
																						noise_var=self.likelihood.noise)

	def predict_gradient_at_test_location_conditioned_on_u(self, test_x_vec, u_vec, sequential=True):
		dmpred_dx_vec, dstdpred_dx_vec = self.model_grad.prediction_special(Xtrain=self.train_inputs[0],
																																				Ytrain=self.train_targets[:,None],
																																				x_in=test_x_vec[:,0,:], # [b x (q=1) x dim] -> # [b x dim]
																																				u_vec=u_vec,
																																				sequential=sequential)
		return dmpred_dx_vec, dstdpred_dx_vec

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return MultivariateNormal(mean_x, covar_x)

	def my_print(self,msg):
		'''
		Place the identifier before any message
		'''
		print("["+self.iden.upper()+"] "+msg)

	def plot(self,axes=None,block=False,Ndiv=100,legend=True,title="GPgrad",plotting=True,plotCDF=False,clear_axes=False,Nsamples=None,ylabel=None,ylim=None,
							pause=None,showtickslabels_x=True):
		'''
		This function hardcodes the plotting limits between zero and one for now
		'''
		if plotting == False or self.dim > 1:
			return

		pp = PlotProbability()
		xpred_vec = torch.linspace(0.0,1.0,Ndiv)[:,None]
		xpred_vec = xpred_vec.unsqueeze(0) # Ndiv batches of [q=1 x self.dim] dimensions each

		# Predict:
		posterior = self.posterior(xpred_vec)

		# Get upper and lower confidence bounds (2 standard deviations from the mean):
		lower_ci, upper_ci = posterior.mvn.confidence_region()

		# Posterior mean:
		mean_vec = posterior.mean

		if self.dim == 1:
			axes = pp.plot_GP_1D(	xpred_vec=xpred_vec.squeeze().cpu().numpy(),
														fpred_mode_vec=mean_vec.squeeze().detach().cpu().numpy(),
														fpred_quan_minus=lower_ci.squeeze().detach().cpu().numpy(),
														fpred_quan_plus=upper_ci.squeeze().detach().cpu().numpy(),
														X_sta=self.train_inputs[0].detach().cpu().numpy(),
														Y_sta=self.train_targets.detach().cpu().numpy(),
														title=title,axes=axes,block=block,
														legend=legend,clear_axes=True,xlabel=None,ylabel=ylabel,xlim=np.array([0.,1.]),ylim=ylim,
														labelsize="x-large",legend_loc="best",colormap="grey",
														showtickslabels_x=showtickslabels_x)

			if Nsamples is not None:
				f_sample = posterior.sample(sample_shape=torch.Size([Nsamples]))
				for k in range(Nsamples):
					axes.plot(xpred_vec.squeeze().detach().cpu().numpy(),
											f_sample[k,0,:,0],linestyle="--",linewidth=1.0,color="sienna")
		
		elif self.dim == 2:
			pass

		plt.show(block=block)
		if pause is not None:
			plt.pause(pause)

		return axes



