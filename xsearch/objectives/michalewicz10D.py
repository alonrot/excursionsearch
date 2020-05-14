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

class Michalewicz10D(ObjectiveFunction):

	def __init__(self):
		'''
		Reported global minimum
		=======================
		Vanaret, Charlie, Jean-Baptiste Gotteland, Nicolas Durand, and Jean-Marc Alliot. 
		"Certified global minima for a benchmark of difficult optimization problems." (2014).
		x_gm = np.array([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796]])
		f_gm = -9.6601517
		domain = [0,pi]^D

		In this paper, the Michalewicz1 parameter is set to m=10, like in our case

		Implementation following GOBench
		================================
		See https://github.com/philipmorrisintl/GOBench/blob/master/gobench/go_benchmark_functions/go_funcs_M.py

		Equations
		=========
		A Literature Survey of Benchmark Functions For Global Optimization Problems
		https://arxiv.org/pdf/1308.4008.pdf

		Visualization (reported global minima unreliable...)
		=============
		http://infinity77.net/global_optimization/test_functions.html#test-functions-index

		'''

		super().__init__(dim=10,noise_std=0.0)

		self.m = 10 # Speed of oscilations
		self.x_gm = np.array([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796]]) / np.pi

	def evaluate(self,x_in,with_noise=True):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		assert x_in.shape[0] == 1
		x_in = x_in.flatten()

		x_in = x_in * np.pi # Domain x_in \in [0,pi]^D

		i = np.arange(1, self.dim + 1)
		aux_vec = np.sin(x_in) * np.sin((x_in ** 2)*i / np.pi) ** (2 * self.m)
		if aux_vec.ndim == 1:
			f_out = -np.sum(aux_vec)
		else:
			f_out = -np.sum(aux_vec,axis=1)

		# Normalize:
		f_out = f_out / 9.6601517 + 0.5 # Normalized to have zero mean and unit variance

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
		x_gm = np.array([[2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796]]) / np.pi
		f_gm = -9.6601517
		f_gm = f_gm / 9.6601517 + 0.5 # Normalized to have zero mean and unit variance
		return x_gm, f_gm
