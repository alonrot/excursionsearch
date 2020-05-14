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

class Simple1D(ObjectiveFunction):

	def __init__(self):
		'''
		'''

		super().__init__(dim=1,noise_std=0.0)

		self.x_gm = np.array([[0.0]]) # Dummy value

	def evaluate(self,x_in,with_noise=True):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		assert x_in.shape[0] == 1

		f_out = 1.0*np.cos(x_in.flatten()*6.*np.pi)*np.exp(-x_in.flatten())

		if with_noise == True:
			y_out = self.add_gaussian_noise(f_out)
		else:
			y_out = f_out

		if isinstance(y_out,np.ndarray):
			assert y_out.ndim == 1
			if y_out.shape[0] == 1:
				y_out = float(y_out[0])

		return y_out

	@staticmethod
	def true_minimum():
		x_gm = np.array([[0.25]])
		f_gm = -0.3
		return x_gm, f_gm
