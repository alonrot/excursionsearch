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
import nlopt
import numpy as np
import torch
import pdb
from xsearch.utils.parsing import get_logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
logger = get_logger(__name__)

class ConstrainedOptimizationNonLinearConstraints():

	def __init__(self,dim,fun_obj,fun_cons,tol_cons=1e-5,tol_x=1e-3,Neval_max_local_optis=200):


		# Safe Constrained problem:
		self.hdl_opt_nonlin_cons = nlopt.opt(nlopt.LN_COBYLA,dim)
		self.hdl_opt_nonlin_cons.set_max_objective(self.fun_obj_wrap)
		self.hdl_opt_nonlin_cons.add_inequality_constraint(self.fun_cons_wrap,tol_cons) # This tolerance is absolute. nlopt assumes g(x) <= 0 as constraint satisfaction
		self.hdl_opt_nonlin_cons.set_lower_bounds(0.0)
		self.hdl_opt_nonlin_cons.set_upper_bounds(1.0)
		# self.hdl_opt_nonlin_cons.set_ftol_rel(tol_acqui*1e-1)
		self.hdl_opt_nonlin_cons.set_xtol_rel(tol_x*1e-1)
		self.hdl_opt_nonlin_cons.set_maxeval(Neval_max_local_optis)

		self.tol_cons_in_opti = 10*tol_cons

		self.fun_obj = fun_obj
		self.fun_cons = fun_cons
		self.dim = dim

	# Interface between pytorch and numpy:
	def fun_obj_wrap(self,x_in, grad_in=np.array([])):
		if not torch.is_tensor(x_in):
			x_in = torch.from_numpy(x_in).to(device=device,dtype=dtype).view((1,1,self.dim))
		val = self.fun_obj(x_in)
		assert len(val) == 1
		return val.item()

	# Interface between pytorch and numpy:
	def fun_cons_wrap(self,x_in, grad_in=np.array([])):
		if not torch.is_tensor(x_in):
			x_in = torch.from_numpy(x_in).to(device=device,dtype=dtype).view((1,1,self.dim))
		val = self.fun_cons(x_in)
		assert len(val) == 1
		return val.item()

	def run_constrained_minimization(self, x_restarts: torch.Tensor):

		logger.info("get_safe_evaluation...")

		# # Get random restarts:
		# x_restarts = get_random_safe_restarts(GPmodel=self.GPcons_list[0],
		# 																			radius=self.radius_influence,
		# 																			Nsamples_per_safe_point=10,
		# 																			Nrestarts_max=self.Nrestarts_safe,
		# 																			delta_t=delta_t,
		# 																			hard_limit_on_max_restarts=True)

		# # Add self.Nrestarts_safe more restarts:
		# x_restarts = np.vstack((x_restarts,np.asarray(self.HaltonSequencer.get(self.Nrestarts_safe))))

		x_restarts = x_restarts.detach().cpu().numpy()
		Nrestarts_local = x_restarts.shape[0]
		x_next_vec = np.zeros((Nrestarts_local,self.dim))
		alpha_val_xnext = np.zeros(Nrestarts_local)
		ind_cons_violated = np.ones(Nrestarts_local,dtype=np.bool) # All initialized to True
		logger.info("Maximizing acquisition function with Nrestarts_local = {0:d}".format(Nrestarts_local))
		for i in range(Nrestarts_local):

			if (i+1) % 10 == 0:
				logger.info("Acquisition function restarted {0:d} / {1:d} times".format(i+1,Nrestarts_local))

			try:
				x_next_vec[i,:] = self.hdl_opt_nonlin_cons.optimize(x_restarts[i])
			except Exception as inst:
				logger.info(type(inst),inst.args)
				alpha_val_xnext[i] = np.nan # Assign NaN to the cases where the optimization fails
			else:

				# Store the optimization value:
				alpha_val_xnext[i] = self.hdl_opt_nonlin_cons.last_optimum_value()

				# Sanity check: If a solution is found, alpha_val_xnext[i] should never be inf, nan or None
				if np.isnan(alpha_val_xnext[i]) or np.isinf(alpha_val_xnext[i]) or alpha_val_xnext[i] is None:
					logger.info("Sanity check failed: The optimizer returns a successful state, but alpha=NaN")

				cons_val = self.fun_cons_wrap(x_next_vec[i])
				if cons_val > self.tol_cons_in_opti:
					logger.info("x_next_vec["+str(i)+"] = "+str(x_next_vec[i,:])+" wasn't a good point because it exceeds the constraint tolerance")
					logger.info("Required cons_val <= "+str(self.tol_cons_in_opti)+", cons_val = "+str(cons_val))
				else:
					ind_cons_violated[i] = False

		# Check for points that violate the constraint:
		if np.all(np.isnan(alpha_val_xnext)):
			logger.info("Something went really wrong here. If this happens, there's no way out...")
			# pdb.set_trace()
			raise ValueError("Fatal error: Optimizer returned NaN in all cases because (i) the acquisition function is corrupted, OR (ii) no feasible solution was found. Abort (!)")
		elif np.all(ind_cons_violated):
			logger.info("No feasible solution was found. Think about increasing the number of random restarts...")
			logger.info("Return the unconstrained minimum...")
			ind_next = np.nanargmin(alpha_val_xnext) # We call nanargmax in case some of the points were exceptions
			x_next = np.atleast_2d(x_next_vec[ind_next])
			alpha_next = alpha_val_xnext[ind_next]
		else:
			logger.info("The constrained problem was succesfully solved!")
			ind_next = np.argmin(alpha_val_xnext[~ind_cons_violated]) # We call nanargmax in case some of the points were exceptions
			x_next = np.atleast_2d(x_next_vec[~ind_cons_violated][ind_next])
			alpha_next = alpha_val_xnext[~ind_cons_violated][ind_next]

		# Get tensors:
		x_next = torch.from_numpy(x_next).to(device=device,dtype=dtype)
		alpha_next = torch.Tensor([alpha_next]).to(device=device,dtype=dtype)

		logger.info("get_safe_evaluation - Done!")
		return x_next, alpha_next

