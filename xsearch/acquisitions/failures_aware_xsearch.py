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
from typing import Optional, List
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective, ScalarizedObjective
from botorch.models.model import Model
from botorch.sampling.samplers import MCSampler
from botorch.utils import t_batch_mode_transform
import numpy as np # These two libraries need to dissapear, as allt he code should be in torch
from scipy.special import erf
from botorch.optim import optimize_acqf
from .acquisition_base import AcquisitionBaseTools
from .acquisition_base_cons import AcquisitionBaseToolsConstrained
from xsearch.utils.get_samples_Frechet import get_fmin_samples_from_gp
import pdb
from .xsearch_unconstrained import Xsearch
from torch.distributions.normal import Normal
from botorch.models import ModelListGP
from scipy.stats import norm
dist_standnormal = Normal(loc=0.0,scale=1.0)
from xsearch.utils.optimize import ConstrainedOptimizationNonLinearConstraints
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.gen import gen_candidates_scipy
from botorch.gen import get_best_candidates

# Used only when inheriting from MCAcquisitionFunction
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
    eta=1e-4,
)

# class XsearchFailures(AnalyticAcquisitionFunction,AcquisitionBaseToolsConstrained):
class XsearchFailures(MCAcquisitionFunction,AcquisitionBaseToolsConstrained):
	def __init__(self, model_list: List[Model], options: dict) -> None:
		
		# AnalyticAcquisitionFunction.__init__(self, model=ModelListGP(model_list[0],model_list[1]), objective=ScalarizedObjective(weights=torch.Tensor([1.0])))
		MCAcquisitionFunction.__init__(self, model=ModelListGP(model_list[0],model_list[1]), objective=constrained_obj)

		AcquisitionBaseToolsConstrained.__init__(self, model_list=model_list, iden="XsearchFailures", 
																										Nrestarts_eta_c=options["Nrestarts_eta_c"],
																										budget_failures=options["budget_failures"])

		self.dim = model_list[0].dim
		self.u_vec = None
		self.Nsamples_fmin = options["Nsamples_fmin"]
		self.Nrestarts_safe = options["Nrestarts_safe"]
		assert self.Nrestarts_safe > 1, "Choose at least 2 restart points."
		self.Nrestarts_risky = options["Nrestarts_risky"]
		assert self.Nrestarts_risky > 1, "Choose at least 2 restart points."
		self.which_mode = "risky"
		self.NBOiters = options["NBOiters"]
		self.rho_conserv = options["rho_safe"]
		self.method_safe = options["method_safe"]
		self.method_risky = options["method_risky"]
		self.constrained_opt = ConstrainedOptimizationNonLinearConstraints(self.dim,self.forward,self.probabilistic_constraint)
		self.use_nlopt = False
		self.disp_info_scipy_opti = options["disp_info_scipy_opti"]
		self.decision_boundary = options["decision_boundary"]

		# Initialize rho latent process:
		if float(self.budget_failures)/self.NBOiters == 1.0:
			self.zk = norm.ppf(self.rho_conserv)
		else:
			self.zk = norm.ppf(float(self.budget_failures)/self.NBOiters)
		self.zrisk = norm.ppf(1.0-self.rho_conserv)
		self.zsafe = norm.ppf(self.rho_conserv)

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

		# Get indices:
		Npred = X.shape[0]
		dim = X.shape[-1]
		Nfcmin_samples = self.u_vec.shape[0]

		# Error checking:
		assert X.dim() in [3,4]
		assert self.u_vec is not None, "Call self.update_u_vec() first"

		dmpred_dx_cond,\
		stdpred_dx_cond = self.model_list[0].predict_gradient_at_test_location_conditioned_on_u(test_x_vec=X, # [b x (q=1) x dim]
																																										u_vec=self.u_vec,
																																										sequential=True)
		# Sanity check:
		assert dmpred_dx_cond.shape[0] == Npred
		assert dmpred_dx_cond.shape[1] == dim
		assert Nfcmin_samples == dmpred_dx_cond.shape[2] == stdpred_dx_cond.shape[2]
		
		# Get posterior GP conditioned on data, to obtain p(u|D)
		posterior_X = self.model_list[0](X)
		# pdf_u_cond_vec = torch.exp(posterior_X.log_prob(self.u_vec.unsqueeze(0).unsqueeze(2)))
		# try:
		u_vec_aux = self.u_vec.view(( [len(self.u_vec)]+[1]*len(X.shape[1::]) )) # Make self.u_vec have the same number of dimensions as X
		pdf_u_cond_vec = torch.exp(posterior_X.log_prob(u_vec_aux))
		pdf_u_cond_vec = pdf_u_cond_vec.view((Nfcmin_samples,Npred)) # REmove the last dimension, if existing
			# pdf_u_cond_vec = torch.exp(posterior_X.log_prob(self.u_vec.unsqueeze(1).unsqueeze(1)))
		# except:
		# 	print("It didn't work... (!)")
		# 	print("X.shape:",X.shape)
		# 	pdf_u_cond_vec = torch.exp(posterior_X.log_prob(self.u_vec.unsqueeze(1)))
		# 	pdb.set_trace()
		# else:
		# print("It worked... (!)")
		# print("X.shape:",X.shape)
		# pdb.set_trace()
		
		pdf_u_cond_vec = pdf_u_cond_vec.T
		try:
			assert pdf_u_cond_vec.shape == torch.Size([Npred,Nfcmin_samples]) # [Npred x Nfcmin_samples]
		except:
			pdb.set_trace()

		alpha_xsearch =	Xsearch.alpha_xsearch(dmpred_dx_cond,stdpred_dx_cond,pdf_u_cond_vec) # 1D vector
		# print("@alpha_xsearch: X.shape: ",X.shape)
		# print("@alpha_xsearch: val.shape: ",val.shape)

		if self.which_mode == "safe" and not self.use_nlopt:
			alpha_xsearch_failures = alpha_xsearch # See eq. (13) in the paper. The probabilistic constraint is taken from self.probabilistic_constraint()

			# When using MCAcquisitionFunction and ConstrainedMCObjective, we can do the constrained optimization in the following way:
			# if len(alpha_xsearch) > 1:
			# 	pdb.set_trace()
			target_obj = alpha_xsearch.view((len(alpha_xsearch),1,1,1))
			target_cons = self.probabilistic_constraint(X).view((len(alpha_xsearch),1,1,1))
			target_both = torch.cat((target_obj,target_cons),dim=-1)
			alpha_xsearch_failures = self.objective(target_both).squeeze(2) # Remove last dimension
			# print("alpha_xsearch_failures.shape:",alpha_xsearch_failures.shape)
		elif self.use_nlopt:
			alpha_xsearch_failures = alpha_xsearch
		else:
			alpha_xsearch_failures = alpha_xsearch * self.get_probability_of_safe_evaluation(X) # See eq. (15) in the paper



		return alpha_xsearch_failures

	def update_remaining_iterations(self,n_iter=0):
		'''
		This function has to be called at each iteration in the outer BO loop
		'''
		assert isinstance(n_iter,int)
		assert n_iter >= 0
		self.Niter_rem = self.NBOiters - n_iter
		assert isinstance(self.Niter_rem,int)
		if self.Niter_rem < 1:
			raise ValueError("Niter_rem must be >= 1")

	def get_probability_of_safe_evaluation(self, X: Tensor) -> Tensor:
		"""
		This function computes Pr(g(x) <= 0) = CDF((0.0 - mu(x))/s(x))
		"""
		posterior_X = self.model_list[1](X)
		return dist_standnormal.cdf(-posterior_X.mean.view(-1)/posterior_X.stddev.view(-1))

	def update_u_vec(self, u_vec: Tensor):
		"""
		u_vec represents is a tensor that contains the samples of the minimum.
		This function is not strictily necessary, as self.u_vec can be set directly, 
		but left for resolution.
		"""
		self.u_vec = u_vec

	def update_latent_process(self):
		"""

		Update underlying dynamic control process, which is used
		to decide between a risky vs. a safe evaluation.
		See Sec. 4.2 in the paper.
		"""

		if self.Niter_rem == self.NBOiters: # We are at the first iteration
			self.my_print("First iteration")
			self.my_print("Update latent rho process: z_{k+1} = z0")
			return self.zk # Which should have been initialized to z0 in self.__init__()

		if self.DeltaBt > self.Niter_rem:
			self.my_print("DeltaBt > DeltaTt || rho_t doesn't play a role, so we set it to the previous value")
			self.my_print("Update latent rho process: z_{k+1} = z_risk")
			return self.zrisk

		if self.DeltaBt == 0:
			self.my_print("DeltaBt = 0 || The budget of failures is gone")
			self.my_print("Update latent rho process: z_{k+1} = z_safe")
			return self.zsafe

		# Get whether the last evaluation was or not a failure:
		# fail_poss = 1.0 - self.GPcons_list[0].get_cdf_untill_thres(np.atleast_2d(self.GPcons_list[0].get_X()[-1,:]),0.0) # using model
		fail_poss = dist_standnormal.cdf(self.model_list[1].train_targets[-1]/torch.sqrt(self.model_list[1].likelihood.noise[0])) # using evaluation

		# Compute controller parameters:
		c_safe = fail_poss/self.DeltaBt # Failing/Not-failing in the context of the remaining budget
		c_risk = self.DeltaBt/self.Niter_rem # Ratio

		# Compute control input:
		uk = (self.zsafe - self.zk)*c_safe + (self.zrisk-self.zk)*c_risk/2.

		# Debug:
		if np.isnan(self.zk):
			pdb.set_trace()

		# Return update:
		return self.zk + uk

	def update_decision_parameter_rho_t(self):
		'''
		update_DeltaBt() needs to be called before this function
		update_remaining_iterations() needs to be called externally (with the current iteration value) before this function
		IMPORTANT: This function must not be called twice in the same iteration
		'''
		self.zk = self.update_latent_process() # Overwrite self.zk with the update

		# Global rho_t: Only actually used in get_safe_evaluation() and get_random_safe_restarts()
		self.rho_t = dist_standnormal.cdf(self.zk)
		self.invCDF_rho_t = self.zk # Same as doing norm.ppf(self.rho_t)

		# self.rho_t = torch.Tensor([0.95]) # debug, remove!
		# self.invCDF_rho_t = dist_standnormal.icdf(self.rho_t) # debug, remove!

		self.my_print("Decision parameter rho_t: {0:2.2f}".format(self.rho_t.item()))

		# Debug:
		if torch.isnan(self.rho_t):
			pdb.set_trace()

	def get_next_point(self) -> (Tensor, Tensor):

		# The following functions need to be called in the given order:
		budget_is_gone = super().update_DeltaBt() # Update remaining budget of failures
		self.update_decision_parameter_rho_t() # Update latent dynamic process
		super().update_eta_c(rho_t=self.rho_conserv) # Update min_x mu(x|D) s.t. Pr(g(x) <= 0) > rho_t

		if budget_is_gone == True:
			self.my_print("Budget of failures depleted. Terminate (!)")
			self.x_next,self.alpha_next = None, None

		# Decide how to explore
		# =====================
		exist_safe_areas = self.does_exist_at_least_one_safe_area(rho_t=self.rho_conserv)
		if exist_safe_areas == True:
			self.my_print("At least one safe area exists!")
			if self.rho_t <= self.decision_boundary and self.DeltaBt > 0.0: # Explore with risk
				self.my_print("=================================================")
				self.my_print("<< Risky exploration: Searching new safe areas >>")
				self.my_print("=================================================")
				self.x_next,self.alpha_next = self.get_risky_evaluation()
			elif self.rho_t > self.decision_boundary and self.DeltaBt > 0.0: # We need to explore safely
				self.my_print("Explore safely, with rho_t = {0:2.2f}".format(self.rho_t.item()))
				self.my_print("===========================================")
				self.my_print("<< Explore safely encountered safe areas >>")
				self.my_print("===========================================")
				self.x_next, self.alpha_next = self.get_safe_evaluation(rho_t=self.rho_t)
			else: # No budget left
				self.my_print("Risky exploration not possible because there's no budget left.")
				self.my_print("Terminate (!)")
				self.x_next, self.alpha_next = None, None
		else:
			if self.DeltaBt == 0.0:
				self.my_print("No safe area has ever been found...")
				self.my_print("Terminate (!)")
				self.x_next, self.alpha_next = None, None
			else:
				self.my_print("Safe area hasn't been found yet...")
				self.my_print("======================================================")
				self.my_print("<< Risky exploration: Searching the first safe area >>")
				self.my_print("======================================================")
				self.x_next, self.alpha_next = self.get_risky_evaluation()

		if self.x_next is not None and  self.alpha_next is not None:
			self.my_print("xnext: " + str(self.x_next.view((1,self.dim)).detach().cpu().numpy()))
			self.my_print("alpha_next: {0:2.2f}".format(self.alpha_next.item()))
		else:
			self.my_print("xnext: None")
			self.my_print("alpha_next: None")

		return self.x_next,self.alpha_next

	def get_risky_evaluation(self):

		# Gather fmin samples, using the Frechet distribution:
		fmin_samples = get_fmin_samples_from_gp(model=self.model_list[0],Nsamples=self.Nsamples_fmin,eta=self.eta_c) # This assumes self.eta has been updated
		self.update_u_vec(fmin_samples)

		self.which_mode = "risky"

		self.my_print("[get_risky_evaluation()] Computing next candidate by maximizing the acquisition function ...")
		options = {"batch_limit": 50,"maxiter": 300,"ftol":1e-9,"method":self.method_risky,"iprint":2,"maxls":20,"disp":self.disp_info_scipy_opti}
		x_next, alpha_next = optimize_acqf(acq_function=self,bounds=self.bounds,q=1,num_restarts=self.Nrestarts_risky,
																			raw_samples=500,return_best_only=True,options=options)
		self.my_print("Done!")

		return x_next, alpha_next

	def get_safe_evaluation(self,rho_t):

		# Gather fmin samples, using the Frechet distribution:
		fmin_samples = get_fmin_samples_from_gp(model=self.model_list[0],Nsamples=self.Nsamples_fmin,eta=self.eta_c) # This assumes self.eta has been updated
		self.update_u_vec(fmin_samples)

		self.which_mode = "safe"

		self.my_print("[get_safe_evaluation()] Computing next candidate by maximizing the acquisition function ...")
		options = {"batch_limit": 50,"maxiter": 300,"ftol":1e-6,"method":self.method_safe,"iprint":2,"maxls":20,"disp":self.disp_info_scipy_opti}
		# x_next, alpha_next = optimize_acqf(acq_function=self,bounds=self.bounds,q=1,num_restarts=self.Nrestarts_safe,raw_samples=500,return_best_only=True,options=options)
		# pdb.set_trace()

		# Get initial random restart points:
		self.my_print("[get_safe_evaluation()] Generating random restarts ...")
		initial_conditions = gen_batch_initial_conditions(acq_function=self,bounds=self.bounds,q=1,
																num_restarts=self.Nrestarts_safe,raw_samples=500, options=options)
		# print("initial_conditions.shape:",initial_conditions.shape)

		# BOtorch does not support constrained optimization with non-linear constraints. Because of this, it provides
		# a work-around solution to optimize using a sigmoid function to push the acquisition function to zero in regions 
		# where the probabilistic constraint is not satisfied (i.e., areas where Pr(g(x) <= 0)) < rho_t.
		self.my_print("[get_safe_evaluation()] Optimizing acquisition function ...")
		x_next_many, alpha_next_many = gen_candidates_scipy(initial_conditions=initial_conditions,
																		acquisition_function=self, lower_bounds=0.0, upper_bounds=1.0, 
																		options=options)
		# Get the best:
		self.my_print("[get_safe_evaluation()] Getting best candidates ...")
		x_next = get_best_candidates(x_next_many, alpha_next_many)

		# pdb.set_trace()
		
		# However, the above optimization does not guarantee that the constraint will be satisfied. The reason for this is that the
		# sigmoid may have a small but non-zero mass in unsafe regions; then, a maximum could be found there in case 
		# the rest of the safe areas are such that the acquisition function is even nearer to zero. If that's the case
		# we trigger a proper non-linear optimizer able to explicitly handle constraints.
		if self.probabilistic_constraint(x_next) > 1e-6: # If the constraint is violated above a tolerance, use nlopt
			self.my_print("[get_safe_evaluation()] scipy optimization recommended an unfeasible point. Re-run using nlopt ...")
			self.use_nlopt = True
			x_next, alpha_next = self.constrained_opt.run_constrained_minimization(initial_conditions.view((self.Nrestarts_safe,self.dim)))
			self.use_nlopt = False
		else:
			self.my_print("[get_safe_evaluation()] scipy optimization finished successfully!")
			alpha_next = self.forward(x_next)
		
		self.my_print("Pr(g(x_next) <= 0): {0:2.8f}".format(self.get_probability_of_safe_evaluation(x_next).item()))		

		# Using botorch optimizer:
		# x_next, alpha_next = optimize_acqf(acq_function=self,bounds=self.bounds,q=1,num_restarts=self.Nrestarts_safe,raw_samples=500,return_best_only=True,options=options)


		# # The code below spits out: Unknown solver options: constraints. Using nlopt instead
		# constraints = [dict(type="ineq",fun=self.probabilistic_constraint)]
		# options = {"batch_limit": 1, "maxiter": 200, "ftol": 1e-6, "method": self.method_risky, "constraints": constraints}
		# x_next,alpha_next = optimize_acqf(acq_function=self,bounds=self.bounds,q=1,num_restarts=self.Nrestarts,
		# 																	raw_samples=500,return_best_only=True,options=options,)

		self.my_print("Done!")

		return x_next, alpha_next

	def probabilistic_constraint(self, X: Tensor) -> Tensor:
		"""
		Pr(g(x) <= 0) > rho

		The above probabilistic constraint is hard to optimize because:
		1) It has a rather short image, i.e., Pr(g(x) <= 0): X -> (0,1)
		2) When the probabily changes abruptly, the gradients become large
		3) If there exist large flat regions where the constraint is clearly satified,
				the gradient tends to zero rapidly.

		To overcome these issues, we propose an equivalent constraint. First we use the 
		cumulative density function of a standard normal distribution:
		Pr(g(x) <= 0) = CDF(-mu(x)/s(x)), where mu(x) = mu(x|D) is the posterior mean given the 
		data D = {Y,X} and s(x) = s(x|D) is the posterior variance of the GP at location x.

		Then, the constraint CDF(-mu(x)/s(x)) >= rho is equivalent to
		mu(x) + s(x)*rho_p <= 0, where rho_p = CDF^{-1}(rho) and CDF^{-1} is the inverse of the
		CDF of a standard normal distribution.
		"""

		mean_x = self.model_list[1](X).mean
		std_x = self.model_list[1](X).stddev
		target_cons = mean_x + self.invCDF_rho_t*std_x

		return target_cons
