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

class Hartmann6D(ObjectiveFunction):

	def __init__(self):
		'''
    Global minimum reported in the GOBench
    ======================================
    self.global_optimum = [[0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]]
    self.fglob = -3.32236801141551
    domain = [0,1]^D

		Implementation following GOBench
		================================
		See https://github.com/philipmorrisintl/GOBench/blob/master/gobench/go_benchmark_functions/go_funcs_H.py

		Equations
		=========
		A Literature Survey of Benchmark Functions For Global Optimization Problems
		https://arxiv.org/pdf/1308.4008.pdf

		Visualization (reported global minima unreliable...)
		=============
		http://infinity77.net/global_optimization/test_functions.html#test-functions-index

		'''

		super().__init__(dim=6,noise_std=0.0)
		

		self.a = np.asarray([[10., 3., 17., 3.5, 1.7, 8.],
												[0.05, 10., 17., 0.1, 8., 14.],
												[3., 3.5, 1.7, 10., 17., 8.],
												[17., 8., 0.05, 10., 0.1, 14.]])

		self.p = np.asarray([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
												[0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
												[0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
												[0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

		self.c = np.asarray([1.0, 1.2, 3.0, 3.2])

		self.x_gm = np.array([[0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]])

	def evaluate(self,x_in,with_noise=True):
		'''
		Overrride to allow for more than single-point evaluation
		'''
		x_in = self.error_checking_x_in(x_in)
		assert x_in.shape[0] == 1

		d = np.sum(self.a * (x_in - self.p) ** 2, axis=1)
		f_out = -np.sum(self.c * np.exp(-d)) 

		# Normalize to have zero mean and unit variance
		f_out = f_out / 3.32236801141551 + 0.5 # Normalized 

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
		x_gm = np.array([[0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054]])
		f_gm = -3.32236801141551
		f_gm = f_gm / 3.32236801141551 + 0.5 # Normalized 
		return x_gm, f_gm
