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
from torch import Tensor
from gpytorch.priors import GammaPrior, SmoothedBoxPrior, NormalPrior
import os
import yaml
import logging

def print_list(list,name,usedim=False):
	print("\n{0:s} list\n=================".format(name))
	np.set_printoptions(precision=3)
	print(get_array_from_list(list,usedim=usedim))
	np.set_printoptions(precision=None)

def get_array_from_list(x_list):

	if isinstance(x_list,list) != True:
		return x_list

	elif len(x_list) == 1 and x_list[0] is None:
		raise ValueError("x_list cannot be an empty list")

	Nel = len(x_list)
	if Nel == 0:
		return None

	arr = np.asarray(x_list)
	if arr.ndim == 2:
		if arr.shape[1] == 1:
			arr = arr[:,0]

	return arr

def print_2Dtensor(tensor: Tensor ,name: str, dim: int) -> None:
	np_array = tensor.view((1,dim)).detach().cpu().numpy()
	np.set_printoptions(precision=3)
	print(np_array)
	np.set_printoptions(precision=None)

def extract_prior(cfg_node: dict, which_type: str):

	assert which_type in ["obj","cons"]

	if cfg_node["lengthscale_prior_type"] == "box": # par1: low, par2: high
		lengthscale_prior = SmoothedBoxPrior(cfg_node["lengthscale_prior_par1_{0:s}".format(which_type)],
																													cfg_node["lengthscale_prior_par2_{0:s}".format(which_type)], sigma=0.001)
	elif cfg_node["lengthscale_prior_type"] == "gamma": # par1: alpha (concentration), par2: beta (rate)
		lengthscale_prior = GammaPrior(	concentration=cfg_node["lengthscale_prior_par1_{0:s}".format(which_type)], 
																										rate=cfg_node["lengthscale_prior_par2_{0:s}".format(which_type)])
	elif cfg_node["lengthscale_prior_type"] == "gaussian":
		lengthscale_prior = NormalPrior(loc=cfg_node["lengthscale_prior_par1_{0:s}".format(which_type)], 	
																										scale=cfg_node["lengthscale_prior_par2_{0:s}".format(which_type)])
	else:
		lengthscale_prior = None
		print("Using no prior for the length scale")

	if cfg_node["outputscale_prior_type"] == "box": # par1: low, par2: high
		outputscale_prior = SmoothedBoxPrior(cfg_node["outputscale_prior_par1_{0:s}".format(which_type)],
																													cfg_node["outputscale_prior_par2_{0:s}".format(which_type)], sigma=0.001)
	elif cfg_node["outputscale_prior_type"] == "gamma": # par1: alpha (concentration), par2: beta (rate)
		outputscale_prior = GammaPrior(	concentration=cfg_node["outputscale_prior_par1_{0:s}".format(which_type)], 
																										rate=cfg_node["outputscale_prior_par2_{0:s}".format(which_type)])
	elif cfg_node["outputscale_prior_type"] == "gaussian":
		outputscale_prior = NormalPrior(loc=cfg_node["outputscale_prior_par1_{0:s}".format(which_type)], 	
																										scale=cfg_node["outputscale_prior_par2_{0:s}".format(which_type)])
	else:
		outputscale_prior = None
		print("Using no prior for the length scale")

	return lengthscale_prior, outputscale_prior
	
def convert_lists2arrays(logvars):

	node2write = dict()
	for key, val in logvars.items():

		if "_list" in key:
			key_new = key.replace("_list","_array")
		else:
			key_new = key

		node2write[key_new] = get_array_from_list(val)

	return node2write


def save_data(node2write: dict, which_obj: str, which_acqui: str, rep_nr: int) -> None:

	# Save data:
	path2obj = "./{0:s}".format(which_obj)
	if not os.path.exists(path2obj):
		print("Creating " + path2obj + " ...")
		os.makedirs(path2obj)

	path2results = path2obj + "/" + which_acqui + "_results"
	if not os.path.exists(path2results):
		print("Creating " + path2results + " ...")
		os.makedirs(path2results)

	path2save = path2results + "/cluster_data"
	if not os.path.exists(path2save):
		print("Creating " + path2save + " ...")
		os.makedirs(path2save)

	file2save = path2save + "/data_" + str(rep_nr) + ".yaml"

	print("\nSaving in {0:s} ...".format(file2save))
	stream_write = open(file2save, "w")
	yaml.dump(node2write,stream_write)
	stream_write.close()

def display_banner(which_algo,Nrep,rep_nr):

	assert which_algo in ["XS","XSF"]

	if which_algo == "XS":
		algo_name = "Excursion Search (XS)"
	if which_algo == "XSF":
		algo_name = "Failures-aware Excursion Search (XSF)"

	banner_name = " <<<<<<<<<<<<<<<<<<< {0:s} >>>>>>>>>>>>>>>>>>> ".format(algo_name)
	line_banner = "="*len(banner_name)

	print(line_banner)
	print(banner_name)
	print(line_banner)
	print(" * Running a total of {0:d} repetition(s)".format(Nrep))
	print(" * Repetition {0:d} / {1:d}".format(rep_nr,Nrep))
	print("")


def get_logger(name):

	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('[%(name)s] %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	return logger


