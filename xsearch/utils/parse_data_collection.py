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
import numpy.linalg as la
import matplotlib.pyplot as plt
import warnings
import sys
import yaml
from datetime import datetime
import os
import pdb
from xsearch.utils.parsing import get_logger
logger = get_logger(__name__)

def generate_folder_at_path(my_path,create_folder=True):

	if my_path is None or my_path == "":
		raise ValueError("my_path must be meaningful...")

	today = datetime.now()
	path2folder = my_path + today.strftime('/%Y%m%d%H%M%S')

	if create_folder == True:
		os.mkdir(path2folder)

	return path2folder

def convert_from_cluster_data_to_single_file(which_obj,which_acqui,Nrepetitions,create_new_folder=True):

	print("")
	logger.info("Parsing collected data into a single file ...")
	logger.info("---------------------------------------------")

	# Error checking:
	algo_list_cons = ["XSF"]
	algo_list_uncons = ["XS"]
	if which_acqui not in algo_list_cons and which_acqui not in algo_list_uncons:
		raise ValueError("which_acqui must be in " + algo_list_uncons + " or in " + algo_list_cons)

	bench_list = ["hart6D","micha10D"]
	if which_obj not in bench_list:
		raise ValueError("which_obj must be in " + bench_list + ", but which_obj = " + which_obj)

	if Nrepetitions < 1 or Nrepetitions > 1000:
		raise ValueError("Check your number of repetitions is correct")

	# Single test regret
	regret_simple_array_list = [None] * Nrepetitions
	DeltaBt_array_list = [None]*Nrepetitions
	rho_t_array_list = [None]*Nrepetitions

	except_vec = np.array([])

	k = -1
	data_corrupted = False
	for i in range(Nrepetitions):

		data_corrupted = False
		# Open corresponding file to the wanted results:
		path2data = "./"+which_obj+"/"+which_acqui+"_results/cluster_data/data_"+str(i)+".yaml"
		logger.info("Loading {0:s} ...".format(path2data))
		try:
			stream 	= open(path2data, "r")
			my_node = (yaml.load(stream,Loader=yaml.Loader)).copy()
		except Exception:
			data_corrupted = True
			logger.info("Data corrupted or non-existent!!!")

		try:
			regret_simple_array_list[k] = my_node['regret_simple_array']
		except Exception:
			logger.info("Some regrets are missing...")
			data_corrupted = True
			pdb.set_trace()

		if np.any(i == except_vec) or data_corrupted == True:
			continue
		else:
			k = k + 1

		if which_acqui in algo_list_cons:
			DeltaBt_array_list[k] = my_node['DeltaBt_array']
			rho_t_array_list[k] = my_node['rho_t_array']

	path4newfolder = "./"+which_obj+"/"+which_acqui+"_results"
	if create_new_folder == True:
		path2save = generate_folder_at_path(path4newfolder)
	else:
		path2save = path4newfolder
	del my_node

	file2save = path2save + "/data.yaml"

	node2write = dict()
	node2write['regret_simple_array_list'] = regret_simple_array_list

	# Add data in particular cases:
	if which_acqui == "XSF":
		node2write['DeltaBt_array_list'] = DeltaBt_array_list
		node2write['rho_t_array_list'] = rho_t_array_list

	logger.info("Saving in {0:s}".format(file2save))
	stream_write = open(file2save, "w")
	yaml.dump(node2write,stream_write)
	stream_write.close()

	# Copy all the cluster data to a folder, as it will be overwritten with subsequent experiments:
	

if __name__ == "__main__":

	if len(sys.argv) != 4:
		raise ValueError("Required input arguments: <ObjFun> <Algorithm> <Nrepetitions> ")

	ObjFun 	= sys.argv[1]
	which_acqui = sys.argv[2]
	Nrepetitions = int(sys.argv[3])

	convert_from_cluster_data_to_single_file(which_obj=ObjFun,which_acqui=which_acqui,Nrepetitions=Nrepetitions)


