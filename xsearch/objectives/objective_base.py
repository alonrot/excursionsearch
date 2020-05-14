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

class ObjectiveFunction():

	def __init__(self,dim,noise_std):

		# Static attributes:
		self._dim = dim
		self._noise_std = noise_std

	@property
	def dim(self):
		return self._dim

	@property
	def noise_std(self):
		return self._noise_std

	def error_checking_x_in(self,x_in):

		if x_in.ndim != 2:
			if x_in.shape[0] == self.dim: # Particular case in which we allow x_in even if it does not have 2 dimensions.
				x_in = np.array([x_in])
			else:
				raise ValueError("x_in does not have the proper size")
		else:
			assert x_in.shape[1] == self.dim

		assert x_in.ndim == 2

		if np.any(np.isnan(x_in.flatten())) == True:
			raise ValueError("x_in contains nans")

		return x_in

	def add_gaussian_noise(self,f_out):
		return f_out + self.noise_std*npr.normal(loc=0.0,scale=1.0)

	def __call__(self,x_in,with_noise=False):
		if torch.is_tensor(x_in):
			x_in = x_in.detach().cpu().numpy()
		
		y_out = self.evaluate(x_in=x_in,with_noise=with_noise)
		y_out = torch.Tensor([y_out])
		return y_out