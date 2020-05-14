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
from gpytorch.means import ConstantMeanGrad
from gpytorch.kernels import ScaleKernel, RBFKernelGrad
from gpytorch.likelihoods import MultitaskGaussianLikelihood
import torch
import pdb
from xsearch.utils.plotting_collection import PlotProbability

class GPmodelWithGrad():
	"""


	This is not a BOtorch model (see that it does not inherit from BOtorch base classes).
	Instead, it's a collection of modules that we use to compute the GP posterior on the
	GP gradient, conditioned on a set of observations. Inheriting from parent classes
	would force us to train the model using both, observations, and observation derivatives; 
	however, that's not what we want.
	"""

	_num_outputs = 1  # to inform GPyTorchModel API

	def __init__(self, dim):
		# squeeze output dim before passing train_Y to ExactGP
		# super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
		# super().__init__(train_X, train_Y, MultitaskGaussianLikelihood(num_tasks=1+train_X.shape[-1]))
		self.likelihood = MultitaskGaussianLikelihood(num_tasks=1+dim)
		self.mean_module = ConstantMeanGrad()
		base_kernel = RBFKernelGrad(ard_num_dims=dim)
		self.covar_module = ScaleKernel(base_kernel=base_kernel)
		# self.to(train_X)  # make sure we're on the right device/dtype
		self.dim = dim

		# # This model is not meant to be trained. Set it always in eval mode
		# self.eval()
		# self.likelihood.eval()

	# def forward(self, x):
	# 	mean_x = self.mean_module(x)
	# 	covar_x = self.covar_module(x)
	# 	# return MultivariateNormal(mean_x, covar_x)
	# 	return MultitaskMultivariateNormal(mean_x, covar_x)

	def update_hyperparameters(self,lengthscale, outputscale, noise_var, mean=0.0):
		self.covar_module.outputscale = outputscale
		self.covar_module.base_kernel.lengthscale = lengthscale
		assert torch.all(noise_var == noise_var[0])
		self.likelihood.noise = noise_var[0]
		self.mean_module.constant[:] = mean

		# Override:
		# self.covar_module.outputscale = 1.0
		# self.covar_module.base_kernel.lengthscale = 0.05

		# print(self.covar_module.outputscale)
		# print(self.covar_module.base_kernel.lengthscale)
		# print(self.likelihood.noise)
		# print(self.mean_module.constant)

	def prediction_special(self, Xtrain, Ytrain, x_in, u_vec, sequential=True):
		"""
		GP derivative prediction, conditioning on the dataset and a virtual noiseless observation {Y,X} U {u,x}
		"""

		if sequential == False:
			raise NotImplementedError("This still needs to be tested...")

		# Error checking:
		NX = Xtrain.shape[0]
		Nx_single = 1
		Nx = x_in.shape[0]
		Nu_samples = u_vec.shape[0]
		assert Nu_samples > 0
		assert u_vec.dim() == 1
		assert not torch.any(torch.isnan(u_vec))
		assert Ytrain.dim() == 2
		assert Ytrain.shape[0] == NX
		assert Ytrain.shape[1] == 1
		assert Xtrain.dim() == 2

		# Concatenate observations with zeroes, which will be replaced later on by u values
		Yall = torch.cat((Ytrain,torch.zeros(size=(Nx_single,1))),dim=0)
		
		dmpred_dx_vec = torch.zeros((Nx,self.dim,Nu_samples))
		dstdpred_dx_vec = torch.zeros((Nx,self.dim,Nu_samples))

		x_sel = torch.zeros(size=(1,self.dim))
		for ii in range(Nx):

			x_sel[:] = x_in[ii,:]

			# Compute joint distribution of p(Y, f*, Df* | X, x*), where X is a set of training inputs and x* a set of test locations
			cov_fXx_fXx, cov_fXx_dfx, cov_dfx_dfx, mean_fXx, mean_dfx = self.prepare_covariance(Xtrain, Ytrain, x_sel)

			# Condition the above distribution and compute \Delta f* | Y, f*, with f* = u
			for jj in range(Nu_samples):

				# Compute predictive mean:
				Yall[NX::,:] = u_vec[jj]
				Kinv_times_Yu = torch.solve((Yall-mean_fXx),cov_fXx_fXx)[0] # torch.solve() returns two elements. We need the first one only
				mpred_df = mean_dfx + torch.matmul(cov_fXx_dfx.T,Kinv_times_Yu)

				# Compute predictive covariance:
				Kinv_times_Kcol = torch.solve(cov_fXx_dfx,cov_fXx_fXx)[0] # torch.solve() returns two elements. We need the first one only
				covpred_df_df = cov_dfx_dfx - torch.matmul(cov_fXx_dfx.T,Kinv_times_Kcol)
				
				# For computing the integral( |Df*|p( Df* | {Y,X} U {u,x} ) ) we only need the variances.
				# Note that the variances are influenced by the data {Y,X} U {u,x}.
				varpred_df_df = covpred_df_df.diag()
				if torch.any(varpred_df_df < 0.0):
					pdb.set_trace()
				
				# Assign, for this value u:
				dmpred_dx_vec[ii,:,jj] = mpred_df.view((-1,self.dim)) # We reshape the piled-up dimensions
				dstdpred_dx_vec[ii,:,jj] = varpred_df_df.sqrt().view(-1,self.dim) # We reshape the piled-up dimensions

		return dmpred_dx_vec, dstdpred_dx_vec

	def prepare_covariance(self, Xtrain, Ytrain, x_in):
		
		"""
		Dimensionality of the covariance matrix
		=======================================

					 Y       f*=u      Df*   [ NX + Nx + self.dim*Nx ] = NX + NX*(1+self.dim)
				__ __ __ __ __ __ __ __ __
			|         |        |        |
		Y |  full   |  full  |  full  |
			|_ _ _ _ _| _ _ _ _| _ _ _ _|
			|         |        |        |
		f*|  full   |  diag  | sparse |
			|_ _ _ _ _| _ _ _ _| _ _ _ _|
			|         |        | block- |
	 Df*|  full   | sparse |  diag  |
			|_ _ _ _ _| _ _ _ _| _ _ _ _|


		where f* = f(x*), Df* = Df(x*), and D denoted "gradient of"

		We are only interested in how the data {Y,X}, together with the virtual observation {u,x*},
		influence the gradient Df* at location x*. We don't need the covariance of f* with Df* 
		at different locations, i.e., 
		
		The reason is that we are in fact ONLY interested in the case in which x* is a single point.
		However, to avoid having an extra loop, we make use of vectorial operations and consider a vector
		x* = (x1*, x2*, ...). Then, we simply set to zero the covariances we are not interested in.
		In fact, considering them would lead to incorrect results. Explicitly, the prior covariances
		
		cov(f(x1*),f(x2*)) = 0
		cov(f(x1*),Df(x2*)) = (0,...,0)
		cov(Df(x1*),Df(x2*)) = (0,...,0)
		
		but, in general, 

		cov(f(x1*),f(x1*)) != 0
		cov(f(x1*),Df(x1*)) != (0,...,0)
		cov(Df(x1*),Df(x1*)) != (0,...,0)

		Thus, dividing the above matrix in 3x3 quadrants, we need the preditive covariance on Df*, 
		as we need it for p(Df* | {Y,X} U {u,x}), which is given by the Gaussian conditioning equations:
		
		cov( Df(x1*),Df(x1*) | {Y,X} U {u,x} ) = Q33 - (Q31 Q32).( (Q11, Q12),(Q21, Q22) )^(-1).(Q13 Q23)

		Getting the above matrix using gpytorch involves index selection, as gpytorch orders the factors in a different way, 
		inconvenient for us. Below, we first slice the covariance matrix provided by gpytorch, then set to zero
		the non-relevant elements, add noise to the observations, and apply the conditioning rule.
		"""

		# Error checking:
		assert x_in.shape[1] == self.dim
		assert x_in.dim() == 2
		assert x_in.shape[0] == 1 # For now
		assert Xtrain.shape[1] == self.dim
		assert Xtrain.dim() == 2
		assert Xtrain.shape[0] > 0
		assert Ytrain.shape[1] == 1
		assert Ytrain.dim() == 2
		assert Ytrain.shape[0] > 0

		# Initialize:
		NX = Xtrain.shape[0]
		Nx = x_in.shape[0]

		# Pile up observations and test locations:
		Xall = torch.cat((Xtrain,x_in),dim=0)

		# Compute prior mean and covariance:
		mean_vec = self.mean_module(Xall)
		C_lazy = self.covar_module(Xall)
		C = C_lazy.evaluate() # Not so expensive if done only once

		# cov_fXx_fXx: ((Q11, Q12),(Q21, Q22))
		ind_fXx = torch.arange(0, (NX+Nx)*(1+self.dim),1+self.dim)
		ind_cov_fXx_fXx_XX, ind_cov_fXx_fXx_YY = torch.meshgrid([ind_fXx, ind_fXx])
		cov_fXx_fXx = C[ind_cov_fXx_fXx_XX,ind_cov_fXx_fXx_YY]

		# Set to zero unneeded cross-correlations, and leave only the variances:
		aux_diag = cov_fXx_fXx[NX::,NX::].diag()
		cov_fXx_fXx[NX::,NX::] = 0.0
		cov_fXx_fXx[NX::,NX::] += aux_diag.diag()
		# pdb.set_trace()

		# Add noise on the observations, and jitter in the predictive locations:
		noise_vec = torch.cat((self.likelihood.noise*torch.ones(NX),1e-5*torch.ones(Nx)))
		cov_fXx_fXx += noise_vec.diag()

		# cov_fXx_dfx: (Q13 Q23)
		ind_dfx = torch.arange(NX*(1+self.dim), (NX+Nx)*(1+self.dim),1) # First, take all the columns
		ind_dfx = ind_dfx.reshape(-1,1+self.dim)[:,1::].reshape(-1) # Now, select those that correspond to dfx
		ind_cov_fXx_dfx_XX, ind_cov_fXx_dfx_YY = torch.meshgrid([ind_fXx, ind_dfx])
		cov_fXx_dfx = C[ind_cov_fXx_dfx_XX,ind_cov_fXx_dfx_YY]
		# print(cov_fX_dfx.shape)

		if Nx > 1: # Eliminate the cross-covariance between fx and dfx:
			# cov_fXx_dfx[NX::] = cov_fXx_dfx[NX::].diag().diag()
			raise NotImplementedError("cov_fXx_dfx is a sparse matrix and needs to be properly computer in self.dim > 1")

		# Sanity check:
		# cov_fXx_dfx_T = C[torch.meshgrid([ind_dfx,ind_fXx])]
		# assert torch.all(cov_fXx_dfx_T == cov_fXx_dfx.T) == True

		# cov_dfx_dfx:
		ind_cov_dfx_dfx_XX, ind_cov_dfx_dfx_YY = torch.meshgrid([ind_dfx, ind_dfx])
		cov_dfx_dfx = C[ind_cov_dfx_dfx_XX,ind_cov_dfx_dfx_YY]
		# print(cov_dfx_dfx.shape)
		if Nx > 1:
			# cov_dfx_dfx = cov_dfx_dfx.diag().diag()
			raise NotImplementedError("cov_dfx_dfx is a block-diagonal matrix and needs to be properly computer in self.dim > 1")
		

		# Get also the means, to have non-zero-mean compatibility:
		mean_fXx = mean_vec[:,0].view((-1,1)) 	# [NX+Nx x 1]
		mean_dfx = mean_vec[NX::,1::].view((-1,1)) 	# [Nx*self.dim x 1] We return the dimensions piled up

		return cov_fXx_fXx, cov_fXx_dfx, cov_dfx_dfx, mean_fXx, mean_dfx

