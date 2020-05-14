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
from xsearch.models.gpmodel import GPmodel

class Simple1DSequentialGP():
    def __init__(self, gp: GPmodel):
        """
        Construct a virtual function, which evaluations are sampled from
        the current GP posterior, and sequentially included in the dataset
        """
        self.gp = gp
        self.x_gm = np.array([[0.0]]) # Dummy value
    def evaluate(self,x_in,with_noise=True):
        return self.gp(x_in).sample(torch.Size([1])).item()
    def true_minimum(self):
        return self.x_gm, -3.0 # Dummy value
    def __call__(self,x_in,with_noise=False):
        return torch.Tensor([self.evaluate(x_in,with_noise)])