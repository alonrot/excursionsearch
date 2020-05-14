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
from .objective_base import ObjectiveFunction

class ConsBallRegions(ObjectiveFunction):

	def __init__(self,dim):
		'''
		g(x) = \prod_{i=1}^D sin(x_i) - t^(-D)
		This function creates 2^(D-1) disjoint unsafe areas. When t = 0, the safe/unsafe areas become perfect hypercubes.

		This function corresponds to the constraint function defined in Sec. 5.2.2 in the paper
		'''

		super().__init__(dim=dim,noise_std=0.0)

	def evaluate(self,x_in,with_noise=True,return_constraint=False,what2read=""):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)

		x_in = x_in*2.*np.pi

		f_out = np.prod(np.sin(x_in),axis=1) - 0.5**self.dim

		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		if isinstance(y_out,np.ndarray):
			assert y_out.ndim == 1
			if y_out.shape[0] == 1:
				y_out = float(y_out[0])

		# Scale it up, to improve the noise2signal ratio:
		# y_out = 10*y_out

		return y_out

def find_roots_of_constraint_in_1D():

	from scipy.optimize import root_scalar
	f = lambda x: np.sin(2*np.pi*x) - 0.5
	x_root1 = root_scalar(f,bracket=[0,0.25],x0=0.1,x1=0.2)
	x_root2 = root_scalar(f,bracket=[0.25,0.5],x0=0.3,x1=0.45)
	print(x_root1) # root: 0.08333333333333334
	print(x_root2) # root: 0.4166666666666667

if __name__ == "__main__":
	find_roots_of_constraint_in_1D()
