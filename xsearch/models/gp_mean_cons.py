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
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.models.model import Model
from botorch.models import FixedNoiseGP, ModelListGP
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
from botorch.acquisition import MCAcquisitionObjective
from typing import Optional, List
import pdb
from torch.distributions.normal import Normal
dist_standnormal = Normal(loc=0.0,scale=1.0)
import warnings

class GPmeanConstrained(MCAcquisitionFunction):
	def __init__(self, 	model: ModelListGP, 
											rho_t: Optional[float] = 0.99,
											objective: Optional[MCAcquisitionObjective] = None) -> None:
		
		super().__init__(model=model, objective=objective)

		self.rho_t = rho_t
		self.rho_t_inv = dist_standnormal.icdf(self.rho_t)
		self.update_infeasible_cost_in_objective()

	def update_infeasible_cost_in_objective(self) -> None:
		"""
		Get infeasible value: We are trying to maximize obj(x) = -mu(x|D), where mu(x|D) is the posterior mean.
		Botorch needs obj(x) to be non-negative. This is done by shifting it up a value M.
		Given the observations, and assuming the model doesn't have ripples, a possible lower bound M
		would be -upper bound on mu(x|D). Such upper bound can be
		the worst observed value plus a CI on the noise uncertainty (assume zero-mean).

		See https://botorch.org/api/_modules/botorch/utils/objective.html#apply_constraints
	
		# Quick reference (copy-pasted):
		This allows feasibility-weighting an objective for the case where the
		objective can be negative by usingthe following strategy:
		(1) add `M` to make obj nonnegative
		(2) apply constraints using the sigmoid approximation
		(3) shift by `-M`

		This function needs to be called every time the models acquire new training points.
		"""
		# Get lowest observation:
		max_obs = torch.max(self.model.subset_output([0]).models[0].train_targets)

		# Accounting for the case in which the worse observation is actually negative
		max_obs_over_mean = Tensor([max(max_obs,0.0)])

		# Subtract 2xnoise_std:
		noise_prior = self.model.subset_output([0]).models[0].likelihood.noise[0]
		upper_bound_on_postmean = max_obs_over_mean + 2.*torch.sqrt(noise_prior)
		self.objective.infeasible_cost = -(-upper_bound_on_postmean) # Flip the sign twice, because the framework maximizes, although we want the minimum
		# print("self.objective.infeasible_cost:",self.objective.infeasible_cost)

	@property
	def rho_t(self):
		return self._rho_t
	
	@rho_t.setter 
	def rho_t(self, rho_t):

		if not torch.is_tensor(rho_t):
			rho_t = Tensor([rho_t])
			
		# Clip the value:
		if rho_t <= 0.0 or rho_t >= 1.0:
			warnings.warn("rho_t cannot take values 0.0 or 1.0 (!)")
			if rho_t >= 1.0: rho_t = 0.99
			if rho_t <= 0.0: rho_t = 0.01
		self._rho_t = rho_t

		# We also set here the CDF inverse of the value:
		self.rho_t_inv = dist_standnormal.icdf(self._rho_t)

	@t_batch_mode_transform(expected_q=1)
	def forward(self, X: Tensor) -> Tensor:
		"""Evaluate scalarized qUCB on the candidate set `X`.

		Args:
		    X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
		        design points each.

		Returns:
				Tensor: A `(b)`-dim Tensor at the given design points `X`.
		"""

		# NOTE: To design this method, we used qNoisyExpectedImprovement as example:
		# https://botorch.org/api/_modules/botorch/acquisition/monte_carlo.html#qNoisyExpectedImprovement
			# posterior = self.model.posterior(X_full)
			# samples = self.sampler(posterior)
			# obj = self.objective(samples)
			# diffs = obj[:, :, :q].max(dim=-1)[0] - obj[:, :, q:].max(dim=-1)[0]
			# return diffs.clamp_min(0).mean(dim=0)

		# NOTE: When using ModelListGP, ModelListGP.__call__ doesn't give the same result as 
		# ModelListGP.posterior; the latter must be used.

		# NOTE: Constraint satisfaction is: g(x) <= 0, see:
		# https://botorch.org/api/_modules/botorch/utils/objective.html#soft_eval_constraint

		prediction = self.model.posterior(X) # This returns botorch.posteriors.gpytorch.GPyTorchPosterior

		# Get the mean and variance of f and g:
		mean_obj 	= prediction.mean[...,0].unsqueeze(-1)
		mean_cons = prediction.mean[...,1].unsqueeze(-1)
		std_obj 	= torch.sqrt(prediction.variance[...,0].unsqueeze(-1))
		std_cons 	= torch.sqrt(prediction.variance[...,1].unsqueeze(-1))

		target_obj = -mean_obj.unsqueeze(0) # Flip the sign because the framework maximizes, although we want the minimum
		target_cons = mean_cons.unsqueeze(0) + self.rho_t_inv*std_cons.unsqueeze(0) # This is lhs of constraint(x) <= 0
		target_both = torch.cat((target_obj,target_cons),dim=-1)

		obj = self.objective(target_both) # No need to flip the sign here
		# print("obj:",obj)
		# print("target_cons:",target_cons)
		# print("target_obj:",target_obj)
		# print("X.shape: ",X.shape)
		# print("@GPmeanConstrained: obj.shape:",obj.shape)

		# pdb.set_trace()

		return obj.squeeze(0)
