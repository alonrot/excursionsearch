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
import pdb

class PlotProbability:
	'''
	Handful of methods to plot 1D and 2D probability densities, and GPs
	===================================================================
	'''

	def __init__(self):

		# Choose colormap:
		# self.change_colormap("blue")

		self.colorbar = None

	def change_plot_attributes_GP_1D(self,colormap=None):

		markersize = 8
		c_opti_color = "xkcd:orange"
		lw_c_opti = 1.5
		color_Xs = "green"
		color_Xu = "red"
		labelsize = "medium" # ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
		if colormap is None:  # Default
			color_mean = 'b'
			color_var = 'b'
		elif colormap == "grey":
			# color_mean = 'silver'
			# color_var 	= 'silver'
			color_mean = 'burlywood'
			color_var 	= 'burlywood'
			c_opti_color = "xkcd:orange"
			markersize = 3	
			lw_c_opti = 0.5
			color_Xs = "black"
			# color_Xu = "black"
			color_Xu = "tomato"
		else:
			print("Specified colormap not implemented yet")
			raise ValueError("colormap must be:\n'grey'\nNone")

		return color_mean,color_var,markersize,c_opti_color,lw_c_opti,color_Xs,color_Xu,labelsize

	def plot_GP_1D(self,xpred_vec,fpred_mode_vec,fpred_quan_minus,fpred_quan_plus,title=None,c_opti=None,thres_implicit=None,X_sta=None,Y_sta=None,
								X_uns=None,axes=None,block=True,colormap=None,legend=True,clear_axes=False,fun_underlying_vec=None,x_underlying_vec=None,xlabel=None,
								ylabel=None,xlim=None,ylim=None,labelsize=None,legend_loc="best",x_bg=None,m_bg=None,showtickslabels_x=True,showtickslabels_y=True,showticks=True,
								plot_c_opti=True):

		if len(np.shape(xpred_vec)) > 1 and np.shape(xpred_vec)[1] > 1:
			print("np.shape(xpred_vec) = ",np.shape(xpred_vec))
			print("plot_GP() method only suited for 1D processes")
			raise NotImplementedError

		if axes is None:
			axes = self.create_axes()

		# Clear axes under demand:
		if clear_axes == True:
			axes.clear()

		# Change colormap:
		color_mean,color_var,markersize,\
		c_opti_color,lw_c_opti,color_Xs,color_Xu,\
		labelsize = self.change_plot_attributes_GP_1D(colormap)

		if showtickslabels_x == False:
			axes.set_xticklabels([])
		if showtickslabels_y == False:
			axes.set_xticklabels([])

		if showticks == False:
			axes.tick_params(
				axis="both",		# changes apply to the x-axis ['x','y','both']
				which="both",		# both major and minor ticks are affected ["major", "minor", "both"]
				bottom=False,		# ticks along the bottom edge are off
				top=False,		# ticks along the top edge are off
				left=False,
				right=False,
				labelbottom=False)		# labels along the bottom edge are off

		# plt.figure(figsize=(10,7))
		# axes.plot(xpred_vec,fpred_mode_vec,color=color_mean,linestyle="-",label="mean",linewidth=2)
		axes.plot(xpred_vec,fpred_mode_vec,color=color_mean,linestyle="-",linewidth=2)
		axes.fill(np.concatenate([xpred_vec, xpred_vec[::-1]]),np.concatenate([fpred_quan_minus,(fpred_quan_plus)[::-1]]),\
			# alpha=.2, fc=color_var, ec='None', label='95% confidence interval')
			alpha=.2, fc=color_var, ec='None')
		if xlim is not None:
			axes.set_xlim(xlim)
		if ylim is not None:
			axes.set_ylim(ylim)
		if xlabel is not None:
			axes.set_xlabel(xlabel,fontsize=labelsize)
		if ylabel is not None:
			axes.set_ylabel(ylabel,fontsize=labelsize)
		if X_sta is not None and Y_sta is not None:
			# axes.plot(X_sta[:,0],Y_sta,marker="o",color=color_Xs,label="Safe evaluation(s)",markersize=markersize,linestyle="None")
			axes.plot(X_sta[:,0],Y_sta,marker="o",color=color_Xs,markersize=markersize,linestyle="None")
		if X_uns is not None:
			if X_sta is not None and Y_sta is not None:
				Xu_level = np.amin(Y_sta)*np.ones(len(X_uns))
			else:
				Xu_level = np.zeros(len(X_uns))
			axes.plot(X_uns,Xu_level,marker="x",color=color_Xu,label="Unsafe evaluation(s)",markersize=1.5*markersize,linestyle="None",markerfacecolor="None")
		if x_bg is not None and m_bg is not None:
			axes.plot(x_bg,m_bg,'mo',label="Global minimum (feasible)",markersize=markersize)
		if c_opti is not None and plot_c_opti == True:
			axes.plot(np.array([xpred_vec[0],xpred_vec[-1]]),c_opti*np.ones(2),linestyle="-",color=c_opti_color,label="c_opti",linewidth=lw_c_opti)
		if thres_implicit is not None:
			axes.plot(np.array([xpred_vec[0],xpred_vec[-1]]),thres_implicit*np.ones(2),linestyle="-",color="xkcd:grey",label="Implicit thres",linewidth=1.0)
		if fun_underlying_vec is not None and x_underlying_vec is not None:
			axes.plot(x_underlying_vec,fun_underlying_vec,'k--',label="True function")
		elif fun_underlying_vec is not None:
			axes.plot(xpred_vec,fun_underlying_vec,'k--',label="True function")
		if legend == True:
			axes.legend(loc=legend_loc)
		if title is not None:
			axes.set_title(title,fontsize=labelsize)
		
		self.show_plot(block=block)

		return axes

	def change_acqui_attributes_1D(self,colormap=None,what2plot="mES_x"):

		x_next_color = "black"
		x_next_marker = "o"
		x_next_marker_size = 4
		axes_linewidth = 0.25
		labelsize="medium" # ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']

		if what2plot == "mES_x":
			color = "silver"
			linestyle = "-"
		elif what2plot == "mESuncons":
			color = "silver"
			linestyle = ":"
		elif what2plot == "cons_vec":
			color = "silver"
			linestyle = "-."
		else:
			color = "silver"
			# color = "gold"
			linestyle = "-"
			# raise ValueError("what2plot must be a string with a valid name")

		return color,linestyle,\
					x_next_color,x_next_marker,x_next_marker_size,\
					axes_linewidth,labelsize

	def plot_acquisition_function(self,var_vec,xpred_vec,x_next=None,acqui_next=None,xlabel=None,ylabel=None,title=None,
																legend=True,axes=None,clear_axes=False,what2plot="mES_x",
																xlim=None,ylim=None,block=False,labelsize=None,showtickslabels=False,showticks=False,
																color=None,label=None):

		if xpred_vec.shape[1] > 1:
			raise NotImplementedError

		if axes is None:
			axes = self.create_axes()

		if clear_axes == True:
			axes.clear()

		if showtickslabels == False:
			axes.set_yticklabels([])
			axes.set_xticklabels([])

		if showticks == False:
			axes.tick_params(
				axis="both",		# changes apply to the x-axis ['x','y','both']
				which="both",		# both major and minor ticks are affected ["major", "minor", "both"]
				bottom=False,		# ticks along the bottom edge are off
				top=False,		# ticks along the top edge are off
				left=False,
				right=False,
				labelbottom=False)		# labels along the bottom edge are off

		# Select plotting attributes depending on what we are plotting:
		color_var_vec,linestyle_var_vec,\
		x_next_color,x_next_marker,x_next_marker_size,\
		axes_linewidth,_labelsize = self.change_acqui_attributes_1D(what2plot=what2plot)

		if labelsize is None:
			labelsize = _labelsize

		# Override the color:
		if color is not None:
			color_var_vec = color

		axes.plot(xpred_vec,var_vec,label=label,color=color_var_vec,linestyle=linestyle_var_vec)

		# for axis_side in ['top','bottom','left','right']:
		#   axes.spines[axis_side].set_linewidth(axes_linewidth)

		if x_next is not None and acqui_next is not None:
			axes.plot(x_next,acqui_next,color=x_next_color,marker=x_next_marker,markersize=x_next_marker_size)
		if title is not None:
			axes.set_title(title,fontsize=labelsize)
		if xlim is not None:
			axes.set_xlim(xlim)
		if ylim is not None:
			axes.set_ylim(ylim)
		if xlabel is not None:
			axes.set_xlabel(xlabel,fontsize=labelsize)
		if ylabel is not None:
			axes.set_ylabel(ylabel,fontsize=labelsize)
		if legend == True:
			axes.legend(fontsize='xx-small')

		self.show_plot(block=block)

		return axes

	def create_axes(self,figsize=None):
		if figsize is not None:
			fig = plt.figure(figsize=figsize)
		else:
			fig = plt.figure(figsize=(14,7))
		axes = fig.add_subplot(1,1,1)
		return axes

	def show_plot(self,block=True):
		'''
		This function blocks the program execution to show a plot, so this fucntion should only be called at the very bottom
		of the code.
		The functions called are only compatible with the TkAgg backend, tipically used in Mac.
		Reference: http://physicalmodelingwithpython.blogspot.com/2015/07/raising-figure-window-to-foreground.html
		'''

		import matplotlib
		matplotlib.use('TkAgg') 
		cfm = plt.get_current_fig_manager()
		# cfm.window.attributes('-topmost', True)
		plt.show(block=block)



# =========================
# Additional plotting tools
# =========================

def plotting_tool_cons(gp_obj,gp_cons,acqui,axes_GPobj,axes_GPcons,axes_acqui,axes_fmin,Ndiv=201,xnext=None,alpha_next=None,iter_nr=None):

  if gp_obj.dim > 1 or gp_cons.dim > 1:
    return None, None

  if axes_GPobj is None and axes_GPcons is None and axes_acqui is None and axes_fmin is None:

    hdl_fig = plt.figure(figsize=(12, 8))
    hdl_fig.suptitle("Failures-aware Excursion Search (XSF)")
    grid_size = (3,5)
    axes_fmin   = plt.subplot2grid(grid_size, (0,0), colspan=1,fig=hdl_fig)
    axes_GPobj  = plt.subplot2grid(grid_size, (0,1), colspan=4,fig=hdl_fig)
    axes_GPcons = plt.subplot2grid(grid_size, (1,1), colspan=4,fig=hdl_fig)
    axes_acqui  = plt.subplot2grid(grid_size, (2,1), colspan=4,fig=hdl_fig)

    # hdl_fig, hdl_axes_obj_splots = plt.subplots(3,1,sharex=False,sharey=False,figsize=))
    # hdl_fig.suptitle("Failures-aware Excursion Search (XSF)")
    # axes_GPobj = hdl_axes_obj_splots[0]
    # axes_GPcons = hdl_axes_obj_splots[1]
    # axes_acqui = hdl_axes_obj_splots[2]
    axes_GPobj = gp_obj.plot(title="Objective f(x)",block=False,axes=axes_GPobj,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=None,showtickslabels_x=False)
    axes_GPcons = gp_cons.plot(title="Constraint g(x)",block=False,axes=axes_GPcons,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=None,showtickslabels_x=False)
    plt.show(block=False)
  else:
    ylim = [-2.5,+2.5]
    # ylim = None
    acqui.plot(axes=axes_acqui,plotting=True,Ndiv=Ndiv,x_next=xnext.detach(),alpha_next=alpha_next.detach(),color="darkgreen",ylabel="alpha(x)",xlabel="x")
    axes_GPobj = gp_obj.plot(title="Objective f(x)",block=False,axes=axes_GPobj,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=None,ylim=ylim,showtickslabels_x=False)
    axes_GPcons = gp_cons.plot(title="Constraint g(x)",block=False,axes=axes_GPcons,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=None,ylim=ylim,showtickslabels_x=False)

    fmin_mean = np.mean(acqui.u_vec.detach().cpu().numpy())
    axes_GPobj.plot([0,1],np.ones(2)*fmin_mean,linestyle="--",color="mediumpurple",linewidth=1.0,label="fmin")
    axes_GPobj.plot(acqui.x_eta_c.squeeze().detach().cpu().numpy(),acqui.eta_c.squeeze().detach().cpu().numpy(),
                    marker="v",markersize=6,color="darkgreen",linestyle="None",label="min_x mu(x|D) s.t. Pr(g(x) <= 0) > 0.99")
    axes_GPcons.plot([0,1],np.zeros(2),linestyle="-",color="grey",linewidth=1.0)

    # Add zones: Tailored for ConsBallRegions with dim=1
    zones_loc = [[0.0,0.084],[0.084,0.417],[0.417,1.0]]
    area_minus = [-3,-3]
    area_plus = [+3,+3]
    axes_GPobj.plot()
    color_var = ["paleturquoise","mediumpurple","paleturquoise"]
    text_list = ["Safe","Unsafe","Safe"]
    for k in range(3):
      axes_GPobj.fill(np.concatenate([zones_loc[k], zones_loc[k][::-1]]),np.concatenate([area_minus,(area_plus)[::-1]]),alpha=.05, fc=color_var[k], ec='None')
      axes_GPcons.fill(np.concatenate([zones_loc[k], zones_loc[k][::-1]]),np.concatenate([area_minus,(area_plus)[::-1]]),alpha=.05, fc=color_var[k], ec='None')
      axes_GPcons.text(sum(zones_loc[k])/2.,3.0,text_list[k],horizontalalignment='center',verticalalignment='center',color=color_var[k])

    axes_GPobj.set_xticks([])
    axes_GPcons.set_xticks([])
    axes_acqui.set_xlabel("x")

    axes_fmin.cla()
    # axes_fmin.hist(acqui.u_vec.detach().cpu().numpy(),bins=5,orientation=u'horizontal',histtype=u'bar', align=u'mid')
    # axes_fmin.hist(acqui.u_vec.detach().cpu().numpy(),bins=5,histtype="barstacked",edgecolor="black",facecolor='mediumpurple',alpha=0.5,color="green",linewidth=1)
    axes_fmin.hist(acqui.u_vec.detach().cpu().numpy(),bins=6,histtype="barstacked",edgecolor="dimgrey",facecolor='cornflowerblue',alpha=0.7,linewidth=1)
    axes_fmin.set_title("fmin samples")
    axes_fmin.set_xlabel("fmin")
    axes_fmin.set_ylabel("count")
    axes_fmin.axvline(x=fmin_mean,color="mediumpurple",linestyle="--")

    # axes_GPobj.legend(loc='lower left')
    plt.show(block=False)
    plt.pause(1.0)

  return axes_GPobj, axes_GPcons, axes_acqui, axes_fmin

def plotting_tool_uncons(gp,acqui,axes_GPobj,axes_acqui,axes_fmin,Ndiv=201,xnext=None,alpha_next=None):

    if gp.dim > 1:
        return None, None

    if axes_GPobj is None and axes_acqui is None:

        hdl_fig = plt.figure(figsize=(12, 8))
        hdl_fig.suptitle("Excursion Search (XS)")
        grid_size = (2,5)
        axes_fmin   = plt.subplot2grid(grid_size, (0,0), colspan=1,fig=hdl_fig)
        axes_GPobj  = plt.subplot2grid(grid_size, (0,1), colspan=4,fig=hdl_fig)
        axes_acqui = plt.subplot2grid(grid_size, (1,1), colspan=4,fig=hdl_fig)

        axes_GPobj = gp.plot(title="Objective f(x)",block=False,axes=axes_GPobj,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=3,showtickslabels_x=False)
    else:
        ylim = [-2.5,+2.5]
        fmin_mean = np.mean(acqui.u_vec.detach().cpu().numpy())
        acqui.plot(axes=axes_acqui,plotting=True,Ndiv=Ndiv,x_next=xnext,alpha_next=alpha_next,color="darkgreen",ylabel="alpha(x)",xlabel="x")
        axes_GPobj = gp.plot(title="Objective f(x)",block=False,axes=axes_GPobj,plotting=True,legend=False,Ndiv=Ndiv,Nsamples=3,ylim=ylim,showtickslabels_x=False)
        axes_GPobj.plot([0,1],np.ones(2)*fmin_mean,linestyle="--",color="mediumpurple",linewidth=1.0,label="fmin")
        axes_GPobj.plot(acqui.x_eta.squeeze().detach().cpu().numpy(),acqui.eta.squeeze().detach().cpu().numpy(),
                    marker="v",markersize=6,color="darkgreen",linestyle="None",label="min_x mu(x|D)")
        axes_GPobj.set_xticks([])

        axes_fmin.cla()
        axes_fmin.hist(acqui.u_vec.detach().cpu().numpy(),bins=6,histtype="barstacked",edgecolor="dimgrey",facecolor='cornflowerblue',alpha=0.7,linewidth=1)
        axes_fmin.set_title("fmin samples")
        axes_fmin.set_xlabel("fmin")
        axes_fmin.set_ylabel("count")
        axes_fmin.axvline(x=fmin_mean,color="mediumpurple",linestyle="--")

        axes_GPobj.legend()
        plt.show(block=False)
        plt.pause(1.0)

    return axes_GPobj, axes_acqui, axes_fmin
