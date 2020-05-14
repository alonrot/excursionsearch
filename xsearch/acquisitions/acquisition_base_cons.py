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
from botorch.models.model import Model
from abc import ABC, abstractmethod
import numpy as np
from xsearch.models.gp_mean import GPmean
from botorch.optim import optimize_acqf
from botorch.gen import gen_candidates_scipy, gen_candidates_torch, get_best_candidates
from botorch.optim.initializers import gen_batch_initial_conditions
from xsearch.utils.plotting_collection import PlotProbability
from botorch.acquisition.objective import ConstrainedMCObjective, ScalarizedObjective, AcquisitionObjective
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from typing import List
import pdb
from botorch.models import FixedNoiseGP, ModelListGP
from xsearch.models.gp_mean_cons import GPmeanConstrained
dist_standnormal = Normal(loc=0.0,scale=1.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idxm = dict(obj=0,cons=1)

def obj_callable(Z):
  return Z[..., 0]

def constraint_callable(Z):
  return 0.0 + Z[..., 1] 	# Z[...,1] represents g(x), with g(x) <= 0 meaning constraint satisfaction.
  												# If we need g(x) >= a, we must return a - Z[..., 1]

# define a feasibility-weighted objective for optimization
constrained_obj = ConstrainedMCObjective(
    objective=obj_callable,
    constraints=[constraint_callable],
    infeasible_cost=0.0,
    eta=1e-3,
)

class AcquisitionBaseToolsConstrained(ABC):

	def __init__(self, model_list: List[Model], iden: str, Nrestarts_eta_c: int, budget_failures: int) -> None:
		"""
		"""

		self.iden = iden
		self.my_print("Starting AcquisitionBaseTools ...")
		self.model_list = model_list

		# # Define GP posterior mean:
		# self.gp_mean_obj = GPmean(self.model_list[idxm['obj']])

		# define models for objective and constraint
		# model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
		# model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con)).to(train_x)
		# combine into a multi-output GP model
		# mll = SumMarginalLogLikelihood(model.likelihood, model)
		# fit_gpytorch_model(mll)

		self.gp_mean_obj_cons = GPmeanConstrained(model=ModelListGP(model_list[0], model_list[1]),
																							objective=constrained_obj)

		# Some options:
		self.Nrestarts_eta_c = Nrestarts_eta_c
		self.budget_failures = budget_failures

		self.dim = self.model_list[idxm['obj']].dim
		self.x_eta_c = None
		self.eta_c = None
		self.bounds = torch.tensor([[0.0]*self.dim, [1.0]*self.dim],device=device)

		# Optimization method: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
		self.method_opti = "L-BFGS-B"
		# self.method_opti = "SLSQP" # constraints
		# self.method_opti = "COBYLA" # constraints


	@abstractmethod
	def get_next_point(self):
		pass

	def update_DeltaBt(self):

		CDF_Xcurr = dist_standnormal.cdf(self.model_list[1].train_targets/torch.sqrt(self.model_list[1].likelihood.noise[0])) # Failures

		Nr_failures = torch.sum(CDF_Xcurr)

		DeltaBt = self.budget_failures - Nr_failures

		# If DeltaBt <= 0.5, even an evaluation exactly in the threshold (i.e., -0.5 failures) will exhaust the budget,
		# so we directly consider it pretty much gone:
		if DeltaBt <= 0.5:
			DeltaBt = torch.Tensor([0.0])

		self.DeltaBt = DeltaBt

		self.my_print("Remaining budget of failures DeltaBt: {0:2.0f}".format(self.DeltaBt.item()))

		return self.budget_failures - Nr_failures <= 0.0

	def get_DeltaBt(self):
		return self.DeltaBt

	def update_eta_c(self, rho_t=0.99):
		"""
		Search the constrained minimum of the posterior mean, i.e.,
		min_x mu(x|D) s.t. Pr(g(x) <= 0) > rho_t
		If no safe area has been found yet, return the best obserbation of f(x) collected so far.
		
		NOTE: Normally, rho_t should be set to the conservative (safe) value, e.g., rho_t = 0.99
		"""

		if self.does_exist_at_least_one_safe_area(rho_t):
			self.x_eta_c, self.eta_c = self.find_eta_c(rho_t)
		else:
			ind_min = torch.argmin(self.model_list[0].train_targets)
			self.x_eta_c = self.model_list[0].train_inputs[0][ind_min,:].view((1,self.dim))
			self.eta_c = self.model_list[0].train_targets[ind_min].view(1)

	def does_exist_at_least_one_safe_area(self,rho_t):
		"""
		Check if at least one of the collected evaluations of the constraint is such that the probabilistic constraint is satisfied.
		If not, we can be sure the constraint is violated everywhere, and it won't make sense to run self.find_eta_c(rho_t)
		
		NOTE: Normally, rho_t should be set to the conservative (safe) value, e.g., rho_t = 0.99
		"""

		exist_safe_areas = torch.any(dist_standnormal.cdf(-self.model_list[1].train_targets/torch.sqrt(self.model_list[1].likelihood.noise[0])) > rho_t)
		return exist_safe_areas

	def find_eta_c(self,rho_t):
		"""
		Find the minimum of the posterior mean, i.e., min_x mu(x|D) s.t. Pr(g(x) <= 0) > rho_t, where D is the data set D={Y,X}, mu(x|D)
		is the posterior mean of the GP queried at location x, and rho_t depends on the current budget of failures.
		"""

		self.my_print("Finding min_x mu(x|D) s.t. Pr(g(x) <= 0) > 0.99")
		self.gp_mean_obj_cons.rho_t = rho_t

		options = {"batch_limit": 1, "maxiter": 200, "ftol": 1e-6, "method": self.method_opti}
		x_eta_c, _ = optimize_acqf(acq_function=self.gp_mean_obj_cons,bounds=self.bounds,q=1,num_restarts=self.Nrestarts_eta_c,
																	raw_samples=500,return_best_only=True,options=options)

		self.my_print("Done!")
		# Revaluate the mean (because the one returned might be affected by the constant that is added to ensure non-negativity)
		eta_c = self.model_list[0](x_eta_c).mean.view(1)

		return x_eta_c, eta_c

	def get_simple_regret_cons(self, fmin_true, function_obj=None):

		ind_safe = self.model_list[1].train_targets <= 0.0
		if len(ind_safe) == 0 or torch.all(ind_safe == False):
			f_simple = torch.max(self.model_list[0].train_targets) # We take the worst observation here. Otherwise, the regret can become non-monotonic
			# ind_min = torch.argmin(self.model_list[0].train_targets)
			# x_simple = self.model_list[0].train_inputs[0][ind_min,:].view((1,self.dim))
		else:
			Ysafe = self.model_list[0].train_targets[ind_safe]
			ind_min_among_safe = torch.argmin(Ysafe)
			f_simple = Ysafe[ind_min_among_safe].view(1)
			# Xsafe = self.model_list[0].train_inputs[0][ind_safe,:].view((1,self.dim))
			# x_simple = Xsafe[ind_min_among_safe,:].view((1,self.dim))

		regret_simple = f_simple - fmin_true

		return regret_simple

	def my_print(self,msg):
		'''
		Place the identifier before any message
		'''
		print("["+self.iden.upper()+"] "+msg)

	def plot(self,axes=None,block=False,title=None,plotting=False,Ndiv=41,showtickslabels=True,
					showticks=True,xlabel=None,ylabel=None,clear_axes=True,legend=False,labelsize=None,normalize=False,
					colorbar=False,color=None,label=None,local_axes=None,x_next=None,alpha_next=None):

		if plotting == False:
			return None

		if self.dim > 1:
			return None

		if local_axes is None and axes is None:
			self.fig,(local_axes) = plt.subplots(1,1,sharex=True,figsize=(10, 7))
		elif local_axes is None:
			local_axes = axes
		elif axes is None:
			pass # If the internal axes already have some value, and no new axes passed, do nothing
		elif local_axes is not None and axes is not None:
			local_axes = axes

		local_pp = PlotProbability()

		if x_next is not None and alpha_next is not None:
			x_next_local = x_next
			alpha_next_local = alpha_next
		else:
			x_next_local = None
			alpha_next_local = 1.0

		test_x_vec = torch.linspace(0.0,1.0,Ndiv)[:,None]
		test_x_vec = test_x_vec.unsqueeze(1) # Make this [Ntest x q x dim] = [n_batches x n_design_points x dim], with q=1 -> Double-check in the documentation!
		var_vec = self.forward(X=test_x_vec).detach().cpu().numpy()

		if self.dim == 1:
			local_axes = local_pp.plot_acquisition_function(var_vec=var_vec,xpred_vec=test_x_vec.squeeze(1),x_next=x_next_local,acqui_next=alpha_next_local,
																			xlabel=xlabel,ylabel=ylabel,title=title,legend=legend,axes=local_axes,clear_axes=clear_axes,
																			xlim=np.array([0.,1.]),block=block,labelsize=labelsize,showtickslabels=showtickslabels,showticks=showticks,
																			what2plot=self.iden,color=color,ylim=None)
			plt.pause(0.25)

		elif self.dim == 2:
			if self.x_next is not None:
				Xs = np.atleast_2d(self.x_next)
			else:
				Xs = self.x_next
			local_axes = local_pp.plot_GP_2D_single(var_vec=var_vec,Ndiv_dim=Ndiv*np.ones(self.dim,dtype=np.int64),Xs=Xs,Ys=self.alpha_next,
													x_label=xlabel,y_label=ylabel,title=title,axes=local_axes,clear_axes=clear_axes,legend=legend,block=block,
													colorbar=colorbar,color_Xs="gold")
			plt.pause(0.25)

		return local_axes


