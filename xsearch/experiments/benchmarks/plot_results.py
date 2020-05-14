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
import matplotlib
# matplotlib.use('TkAgg') # Solves a no-plotting issue for macOS users
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
import pdb
import sys
import yaml
import os

# List of algorithms:
list_algo_uncons = ["XS"]
list_algo_cons = ["XSF"]

# Attributes:
color_mean_dict = dict(EIC="goldenrod",PESC="sienna",RanCons="mediumpurple",XSF="darkgreen")
color_mean_dict.update(dict(EI="goldenrod",PES="sienna",Ran="mediumpurple",XS="darkgreen",mES="lightcoral",PI="cornflowerblue",UCB="grey"))
marker_dict = dict(EIC="v",PESC="o",RanCons="*",XSF="s")
marker_dict.update(dict(EI="v",PES="o",Ran="*",XS="s",mES="D",PI="P",UCB="."))
labels_dict = dict(EIC="EIC",PESC="PESC",RanCons="RanCons",XSF="XSF",XS="XS",EI="EI",PES="PES",Ran="Ran",mES="mES",PI="PI",UCB="UCB")

def add_errorbar(mean_vec,std_vec,axis,ylabel=None,color=None,subtle=False,label_legend=None,
								plot_every=1,marker=None,linestyle=None,linewidth=1,capsize=4,marker_every=1,
								xlabel=None,markersize=5):

	# if var_vec is None:
	# 	return axis
	assert isinstance(mean_vec,np.ndarray)
	mean_vec = mean_vec.flatten()
	assert mean_vec.ndim == 1
	Npoints = mean_vec.shape[0]
	assert Npoints > 0

	if color is None:
		color = npr.uniform(size=(3,))

	# Input points:
	if plot_every is not None:
		if plot_every >= Npoints/2:
			plot_every = 1
	else:
		plot_every = 1

	x_vec = np.arange(1,Npoints+1)
	ind_plot_here = x_vec % plot_every == 0
	x_vec_sep = x_vec[ind_plot_here]
	mean_vec_sep = mean_vec[ind_plot_here]
	std_vec_sep = std_vec[ind_plot_here]

	# Plot:
	if subtle == True:
		axis.plot(x_vec,mean_vec,linestyle=":",linewidth=0.5,color=color)
	else:
		axis.errorbar(x=x_vec,y=mean_vec,yerr=std_vec/2.,marker=marker,linestyle=linestyle,
									linewidth=linewidth,color=color,errorevery=plot_every,capsize=capsize,
									markevery=marker_every,label=label_legend,markersize=markersize)
	# axis.plot(x_vec,mean_vec,marker=marker,linestyle=linestyle,linewidth=linewidth,color=color,label=label_legend)
	if xlabel is not None:
		axis.set_xlabel(xlabel)
	if ylabel is not None:
		axis.set_ylabel(ylabel)
	return axis

def add_plot(var_vec,axis,ylabel=None,color=None,subtle=False):

	# if var_vec is None:
	# 	return axis

	assert isinstance(var_vec,np.ndarray)
	var_vec = var_vec.flatten()
	assert var_vec.ndim == 1
	Npoints = var_vec.shape[0]
	assert Npoints > 0

	if color is None:
		color = npr.uniform(size=(3,))

	# Input points:
	x_vec = range(1,Npoints+1)

	# Plot:
	if subtle == True:
		axis.plot(x_vec,var_vec,linestyle=":",linewidth=0.5,color=color)
	else:
		axis.plot(x_vec,var_vec,marker=".",linestyle="--",linewidth=1,color=color)
	# axis.set_xlabel("Nr. iters")
	if ylabel is not None:
		axis.set_ylabel(ylabel)
	return axis

def compute_mean_and_var(array_list,pop_unwanted_els=False,pop_N=None,hold_after_termination=True,NBOiters_max=None,get_log_data=False):

	Nels = len(array_list)

	array_list_copy = list(array_list)

	assert Nels > 0
	assert isinstance(array_list_copy[0],np.ndarray)
	if array_list_copy[0].ndim > 1: # Flatten
		for k in range(Nels):
			assert np.sum(array_list_copy[k].shape != 1) == 1 # Ensure 1D or 2D column or row vector
			array_list_copy[k] = array_list_copy[k].flatten()

	# Check that all dimensions are the same
	NBOiters_vec = np.zeros(Nels,dtype=int)
	for k in range(Nels):
		NBOiters_vec[k] = len(array_list_copy[k])

	if pop_unwanted_els == True:
		if pop_N is None:
			Nrequested_length = NBOiters_max
		else:
			Nrequested_length = pop_N

		N_wanted_els = np.sum(Nrequested_length == NBOiters_vec)
		assert N_wanted_els > 0
		c = 0
		while c < len(array_list_copy):
			if len(array_list_copy[c]) != Nrequested_length:
				array_list_copy.pop(c)
			else:
				c += 1
		assert len(array_list_copy) == N_wanted_els
	else:
		if NBOiters_max is None:
			NBOiters_max = np.max(NBOiters_vec)
		else:
			NBOiters_vec[NBOiters_vec > NBOiters_max] = NBOiters_max

		if hold_after_termination == True: # Append the last value until NBOiters_max is reached for those experiments that finished beforehand
			print("Hold after termination...")
			ind_maxBOiters_not_equal = NBOiters_max != NBOiters_vec
			if np.any(ind_maxBOiters_not_equal):
				list_pos_NBOiters_shorter = np.arange(Nels)[ind_maxBOiters_not_equal]
				for k in list_pos_NBOiters_shorter:
					# array_list_copy[k] = array_list_copy[k][0:]
					array_list_copy[k] = np.append(array_list_copy[k],np.ones(NBOiters_max-NBOiters_vec[k])*array_list_copy[k][-1])
			# else:
			# 	print("Nothing to modify...")
		else:
			if np.any(NBOiters_max != NBOiters_vec): # Cut the list
				print("NBOiters_vec:",NBOiters_vec)
				NBOiters = np.amin(NBOiters_vec)
				for k in range(Nels):
					array_list_copy[k] = array_list_copy[k][0:NBOiters]
				print("The list had to be cut from "+str(NBOiters_max)+" to "+str(NBOiters)+" !!!")
			else:
				NBOiters = NBOiters_max

	my_array = np.asarray(array_list_copy) # Vertically stacks 1-D vectors
	if NBOiters_max is not None:
		my_array = my_array[:,0:NBOiters_max]

	if get_log_data == True:
		my_array = np.log10(my_array)

	mean_vec = np.mean(my_array,axis=0)
	std_vec = np.std(my_array,axis=0)

	assert mean_vec.ndim == 1
	assert std_vec.ndim == 1

	return mean_vec,std_vec

def get_plotting_data(which_obj,which_acqui,nr_exp,save_plot,block=True,pop_unwanted_els=True,NBOiters_max=None,
						get_DeltaBt=False,get_log_data=False,alternative_path=None,log_transf=False):

	# Error checking:
	acqui_list = ["XS","XSF"]
	if which_acqui not in acqui_list:
		raise ValueError("which_acqui must be in " + str(acqui_list))

	# Open corresponding file to the wanted results:
	path2data = "./{0:s}/{1:s}_results/{2:s}/data.yaml".format(which_obj,which_acqui,nr_exp)
	print("Loading {0:s} ...".format(path2data))
	stream 	= open(path2data, "r")
	my_node = yaml.load(stream,Loader=yaml.Loader)
	stream.close()

	Nrepetitions = len(my_node["regret_simple_array_list"])
	regret_simple_array_list = my_node['regret_simple_array_list']
	if which_acqui == "XSF":
		DeltaBt_array_list = my_node['DeltaBt_array_list']
		rho_t_array_list = my_node['rho_t_array_list']
	else:
		DeltaBt_array_list = None
		rho_t_array_list = None

	hdl_fig, hdl_splot = plt.subplots(3,1,figsize=(9,9))
	hdl_plt_regret_simple = hdl_splot[0]
	hdl_plt_DeltaBt = hdl_splot[1]
	hdl_plt_delta_t = hdl_splot[2]
	hdl_plt_delta_t.set_xlabel("Iteration")
	hdl_plt_DeltaBt.set_xticks([])
	hdl_plt_regret_simple.set_xticks([])
	hdl_plt_regret_simple.grid(which="major",axis="both")

	for k in range(Nrepetitions):

		# Empirical Regret v1:
		if isinstance(regret_simple_array_list[k],np.ndarray) == True and regret_simple_array_list[k] is not None:
			add_plot(var_vec=regret_simple_array_list[k],axis=hdl_plt_regret_simple,ylabel="Simple Regret",subtle=True,color="cornflowerblue")

		# DeltaBt:
		if which_acqui == "XSF":
			if isinstance(DeltaBt_array_list[k],np.ndarray) == True and DeltaBt_array_list[k] is not None:
				add_plot(var_vec=DeltaBt_array_list[k],axis=hdl_plt_DeltaBt,ylabel="DeltaBt",subtle=True,color="cornflowerblue")

			if isinstance(rho_t_array_list[k],np.ndarray) == True and rho_t_array_list[k] is not None:
				add_plot(var_vec=rho_t_array_list[k],axis=hdl_plt_delta_t,ylabel="delta_t",subtle=True,color="cornflowerblue")

	regret_simple_mean,regret_simple_std = compute_mean_and_var(regret_simple_array_list,hold_after_termination=True,NBOiters_max=NBOiters_max,get_log_data=get_log_data)
	add_plot(var_vec=regret_simple_mean,axis=hdl_plt_regret_simple,color="lightcoral")

	# Add DeltaBt:
	if which_acqui == "XSF":
		DeltaBt_mean, DeltaBt_std = compute_mean_and_var(DeltaBt_array_list,hold_after_termination=True, NBOiters_max=NBOiters_max)
		add_plot(var_vec=DeltaBt_mean,axis=hdl_plt_DeltaBt,color="lightcoral")

		rho_t_mean, rho_t_std = compute_mean_and_var(rho_t_array_list,hold_after_termination=True, NBOiters_max=NBOiters_max)
		add_plot(var_vec=rho_t_mean,axis=hdl_plt_delta_t,color="lightcoral")
	else:
		DeltaBt_mean, DeltaBt_std, rho_t_mean, rho_t_std = None, None, None, None

	if save_plot == True:
		print("Saving plot...")
		hdl_fig.tight_layout()
		path2save_figure = "./"
		file_name = "tmp_"+which_acqui+"_"+str(dim)
		plt.savefig(path2save_figure+file_name)
		print("Saved!")
	elif block == True:
		plt.show(block=True)
	else:
		plt.close(hdl_fig)
		return regret_simple_mean, regret_simple_std, DeltaBt_mean, DeltaBt_std, rho_t_mean, rho_t_std

def add_plot_attributes(axes,fontsize_labels,ylabel,xlabel=None,supress_xticks=False):
	if xlabel is not None:
		axes.set_xlabel(xlabel,fontsize=fontsize_labels+2)
	axes.set_ylabel(ylabel,fontsize=fontsize_labels+2)
	if supress_xticks == True:
		axes.set_xticklabels([])
	else:
		axes.tick_params('x',labelsize=fontsize_labels)
	axes.tick_params('y',labelsize=fontsize_labels)
	axes.grid(b=True,which="major",color='grey', linestyle=':', linewidth=0.5)
	return axes

def plot(which_obj,which_acqui,nr_exp):

	# Get the experiment number if None passed:
	if nr_exp is None:
		path2data = "./{0:s}/{1:s}_results/".format(which_obj,which_acqui)

		# From all the folders that start with '2020' take the largest number (most recent):
		dir_list = os.listdir(path2data)
		name_most_recent = "0"
		for k in range(len(dir_list)):
			if "20" in dir_list[k]:
				if int(dir_list[k]) > int(name_most_recent):
					name_most_recent = dir_list[k]

		if name_most_recent == "0":
			raise ValueError("No experiment found (!)")

		nr_exp = name_most_recent

	# List of algorithms:
	is_constrained = False
	if which_acqui in list_algo_cons:
		is_constrained = True
		list_algo = list_algo_cons
	else:
		list_algo = list_algo_uncons

	title_simp = "Simple regret"
	fontsize_labels = 18
	figsize = (9,8)

	if is_constrained:

		grid_total = (4,1)
		grid_simp = grid_inf = (0,0)
		grid_DeltaBt_inf = grid_DeltaBt_simp = (2,0)
		grid_delta_t_inf = grid_delta_t_simp = (3,0)
		xlabel_DeltaBt = ""
		xlabel_delta_t = "Iteration"
		supress_xticks_DeltaBt = True
		supress_xticks_delta_t = False
		
		# General plotting settings:
		plt.rc('font', family='serif')
		plt.rc('legend',fontsize=fontsize_labels+2)

		hdl_fig_simp = plt.figure(figsize=figsize)
		hdl_splot_simp 		= plt.subplot2grid(grid_total, grid_simp, rowspan=2,fig=hdl_fig_simp)
		hdl_splot_DeltaBt_simp = plt.subplot2grid(grid_total, grid_DeltaBt_simp, rowspan=1,fig=hdl_fig_simp)

		hdl_splot_simp = add_plot_attributes(hdl_splot_simp,fontsize_labels,title_simp,supress_xticks=True)
		hdl_splot_DeltaBt_simp = add_plot_attributes(hdl_splot_DeltaBt_simp,fontsize_labels,"$\Delta B_t$",
															xlabel=xlabel_DeltaBt,supress_xticks=supress_xticks_DeltaBt)

		hdl_splot_simp.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		hdl_splot_delta_t_simp = plt.subplot2grid(grid_total, grid_delta_t_simp, rowspan=1,fig=hdl_fig_simp)
		hdl_splot_delta_t_simp = add_plot_attributes(hdl_splot_delta_t_simp,fontsize_labels,"$\\rho_t$",xlabel=xlabel_delta_t,
														supress_xticks=supress_xticks_delta_t)
	else:

		plt.rc('font', family='serif')
		plt.rc('legend',fontsize=fontsize_labels+2)
		hdl_fig_simp, hdl_splot_simp = plt.subplots(1,1,sharex=False,sharey=False,figsize=figsize)
		hdl_splot_simp = add_plot_attributes(hdl_splot_simp,fontsize_labels,title_simp,xlabel="Iteration")

	plot_every = 10
	marker_every = 5
	linestyle = "-"
	linewidth = 1
	capsize = 4
	str_table_list = [None]*len(list_algo)

	for i in range(len(list_algo)):

		regret_simple_mean,\
		regret_simple_std,\
		DeltaBt_mean,\
		DeltaBt_std,\
		rho_t_mean,\
		rho_t_std = get_plotting_data(which_obj,list_algo[i],nr_exp,save_plot=False,block=False)

		add_errorbar(regret_simple_mean,regret_simple_std,axis=hdl_splot_simp,color=color_mean_dict[list_algo[i]],label_legend=labels_dict[list_algo[i]],
									plot_every=plot_every,marker=marker_dict[list_algo[i]],linestyle=linestyle,linewidth=linewidth,capsize=capsize,marker_every=marker_every)
	
		if is_constrained:
			add_errorbar(DeltaBt_mean,DeltaBt_std,axis=hdl_splot_DeltaBt_simp,color=color_mean_dict[list_algo[i]],label_legend=labels_dict[list_algo[i]],
									plot_every=plot_every,marker=marker_dict[list_algo[i]],linestyle=linestyle,linewidth=linewidth,capsize=capsize,marker_every=marker_every)
			add_errorbar(rho_t_mean,rho_t_std,axis=hdl_splot_delta_t_simp,color=color_mean_dict[list_algo[i]],label_legend=labels_dict[list_algo[i]],
									plot_every=plot_every,marker=marker_dict[list_algo[i]],linestyle=linestyle,linewidth=linewidth,capsize=capsize,marker_every=marker_every)

	# How to change xticks:
	hdl_splot_simp.legend()

	plt.show(block=True)


if __name__ == "__main__":

	if len(sys.argv) not in [3,4]:
		raise ValueError("Required input arguments: <ObjFun> <which_acqui> [<nr_exp>]")
	ObjFun = sys.argv[1]
	which_acqui = sys.argv[2]
	if len(sys.argv) == 4:
		nr_exp = sys.argv[3]
	else:
		nr_exp = None

	plot(which_obj=ObjFun,which_acqui=which_acqui,nr_exp=nr_exp)





