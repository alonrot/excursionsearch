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
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from xsearch.acquisitions import XsearchFailures
from xsearch.objectives import Hartmann6D, Michalewicz10D, Simple1D, ConsBallRegions
from xsearch.models.gpmodel import GPmodel
from botorch.utils.sampling import draw_sobol_samples
from xsearch.utils.parsing import convert_lists2arrays, save_data, display_banner, get_logger
from xsearch.utils.plotting_collection import plotting_tool_cons
import yaml
from xsearch.utils.parse_data_collection import convert_from_cluster_data_to_single_file
logger = get_logger(__name__)
np.set_printoptions(linewidth=10000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def initialize_logging_variables():
    logvars = dict( mean_bg_list=[],
                    x_bg_list=[],
                    x_next_list=[],
                    alpha_next_list=[],
                    Xevals_array=[],
                    Yevals_obj_array=[],
                    Yevals_cons_array=[],
                    regret_simple_list=[],
                    Xinit=0,
                    Yinit_obj=0,
                    Yinit_cons=0,
                    DeltaBt_list=[],
                    rho_t_list=[])
    return logvars

def append_logging_variables(logvars,eta_c,x_eta_c,x_next,alpha_next,regret_simple,DeltaBt,rho_t):
    logvars["mean_bg_list"].append(eta_c.view(1).detach().cpu().numpy())
    logvars["x_bg_list"].append(x_eta_c.view(x_eta_c.shape[1]).detach().cpu().numpy())
    logvars["x_next_list"].append(x_next.view(x_eta_c.shape[1]).detach().cpu().numpy())
    logvars["alpha_next_list"].append(alpha_next.view(1).detach().cpu().numpy())
    logvars["regret_simple_list"].append(regret_simple.view(1).detach().cpu().numpy())
    logvars["DeltaBt_list"].append(DeltaBt.view(1).detach().cpu().numpy())
    logvars["rho_t_list"].append(rho_t.view(1).detach().cpu().numpy())
    return logvars

def get_initial_evaluations(which_objective,function_obj,function_cons):

    assert which_objective in ["hart6D", "micha10D","simple1D"], "Objective function <which_objective> must be {'hart6D','micha10D','simple1D'}"

    # Get initial evaluation:
    if which_objective == "hart6D":
        train_x = torch.Tensor([[0.32124528, 0.00573107, 0.07254258, 0.90988337, 0.00164314, 0.41116992]]) # Randomly computed
    
    if which_objective == "micha10D":
        train_x = torch.Tensor([[0.65456088, 0.22632844, 0.50252072, 0.80747863, 0.11509346, 0.73440179, 0.06093292, 0.464906, 0.01544494, 0.90179168]]) # Randomly computed

    # Get initial evaluation:
    if which_objective == "simple1D":
        train_x = draw_sobol_samples(bounds=torch.Tensor(([0.0],[1.0])),n=1,q=1).squeeze(1)

    # Get initial evaluation in f(x):
    train_y_obj = function_obj(train_x)

    # Get initial evaluation in g(x):
    train_y_cons = function_cons(train_x)
    return train_x, train_y_obj, train_y_cons

def get_function_obj(which_objective):

    assert which_objective in ["hart6D", "micha10D","simple1D"], "Objective function <which_objective> must be {'hart6D','micha10D','simple1D'}"

    if which_objective == "hart6D":
        func_obj = Hartmann6D()
        dim = 6
    if which_objective == "micha10D":
        func_obj = Michalewicz10D()
        dim = 10
    if which_objective == "simple1D":
        func_obj = Simple1D()
        dim = 1

    # Get the true minimum for computing the regret:
    x_min, f_min = func_obj.true_minimum()
    logger.info("<<< True minimum >>>")
    logger.info("====================")
    logger.info("  x_min:" + str(x_min))
    logger.info("  f_min:" + str(f_min))

    return func_obj, dim, x_min, f_min

def get_acquisition_function(which_algo):
    logger.info("Initializing {0:s} ...".format(which_algo))
    if which_algo == "XSF":
        AcquiFun = XsearchFailures
    return AcquiFun

def run(rep_nr,which_algo):

    # Load configuration file and store it in a global dictionary:
    config_file_path = "./conf_bench.yaml"
    stream  = open(config_file_path, "r")
    cfg_node = yaml.load(stream,Loader=yaml.Loader)
    stream.close()

    # Override some options when toy mode is activated:
    if cfg_node["toy_mode"]:
        cfg_node["which_objective"] = "simple1D"
        cfg_node["NBOiters"] = 8
        cfg_node["budget_failures"] = 3
        cfg_node["Nrestarts_safe"] = 4
        cfg_node["Nrestarts_risky"] = 4
        cfg_node["lengthscale_prior_type"] = "box"
        cfg_node["lengthscale_prior_par1_obj"] = 0.01
        cfg_node["lengthscale_prior_par2_obj"] = 0.3
        cfg_node["lengthscale_prior_par1_cons"] = 0.01
        cfg_node["lengthscale_prior_par2_cons"] = 0.3
    elif cfg_node["thesis_plot_mode"] == True:
        cfg_node["which_objective"] = "simple1D"
        cfg_node["NBOiters"] = 15
        cfg_node["budget_failures"] = 3
        cfg_node["Nrestarts_safe"] = 4
        cfg_node["Nrestarts_risky"] = 4
        cfg_node["lengthscale_prior_type"] = "box"
        cfg_node["lengthscale_prior_par1_obj"] = 0.01
        cfg_node["lengthscale_prior_par2_obj"] = 0.3
        cfg_node["lengthscale_prior_par1_cons"] = 0.01
        cfg_node["lengthscale_prior_par2_cons"] = 0.3
    else:
        cfg_node["plotting"] = False

    # Random seed for numpy and torch:
    np.random.seed(rep_nr)
    torch.manual_seed(rep_nr)

    # Load true function and initial evaluations:
    function_obj, dim, x_min, f_min = get_function_obj(cfg_node["which_objective"])
    
    # Constraint function:
    function_cons = ConsBallRegions(dim=dim)

    train_x, train_y_obj, train_y_cons = get_initial_evaluations(cfg_node["which_objective"],function_obj,function_cons)

    # Load GP model and fit hyperparameters:
    gp_list = [ GPmodel(train_X=train_x, train_Y=train_y_obj, options=cfg_node, which_type="obj"),
                GPmodel(train_X=train_x, train_Y=train_y_cons, options=cfg_node, which_type="cons")]
    gp_list[0].update_hyperparameters()
    gp_list[0].display_hyperparameters()
    gp_list[1].update_hyperparameters()
    gp_list[1].display_hyperparameters()

    # Initialize acquisition function:
    AcquiFun = get_acquisition_function(which_algo)
    acqui = AcquiFun(model_list=gp_list, options=cfg_node)

    logvars = initialize_logging_variables()
    logvars["Xinit"]    = train_x.detach().cpu().numpy()
    logvars["Yinit_obj"]   = train_y_obj.detach().cpu().numpy()
    logvars["Yinit_cons"]   = train_y_cons.detach().cpu().numpy()

    # Plotting:
    if cfg_node["plotting"]:
        axes_GPobj, axes_GPcons, axes_acqui, axes_fmin = plotting_tool_cons(gp_list[0],gp_list[1],acqui,axes_GPobj=None,axes_GPcons=None,axes_acqui=None,axes_fmin=None)

    # average over multiple trials
    if cfg_node["thesis_plot_mode"] == True:
        rho_t_vec = np.zeros(cfg_node["NBOiters"])
        zk_t_vec = np.zeros(cfg_node["NBOiters"])
        DeltaBt_vec = np.zeros(cfg_node["NBOiters"])
        y_cons_vec = np.zeros(cfg_node["NBOiters"])
    
    for trial in range(cfg_node["NBOiters"]):
        
        msg_bo_iters = " <<< BO Iteration {0:d} / {1:d} >>>".format(trial+1,cfg_node["NBOiters"])
        print("\n")
        logger.info("="*len(msg_bo_iters))
        logger.info("{0:s}".format(msg_bo_iters))
        logger.info("="*len(msg_bo_iters))

        # Get next point:
        acqui.update_remaining_iterations(n_iter=trial)
        xnext, alpha_next = acqui.get_next_point()

        # Do this thing:
        if cfg_node["thesis_plot_mode"] == True:
            rho_t_vec[trial] = acqui.rho_t.item()
            zk_t_vec[trial] = acqui.zk
            DeltaBt_vec[trial] = acqui.DeltaBt.item()
            y_cons_vec[trial] = gp_list[1].train_targets[-1].item()

        # Compute simple regret:
        regret_simple = acqui.get_simple_regret_cons(fmin_true=f_min)
        logger.info("Regret: {0:2.5f}".format(regret_simple.item()))
        
        if xnext is None and  alpha_next is None:
            break
        
        # Logging:
        if cfg_node["plotting"]:
            axes_GPobj, axes_GPcons, axes_acqui, axes_fmin = plotting_tool_cons(gp_list[0],gp_list[1],acqui,axes_GPobj,axes_GPcons,axes_acqui,axes_fmin,xnext=xnext,alpha_next=alpha_next)
        append_logging_variables(logvars,acqui.eta_c,acqui.x_eta_c,xnext,alpha_next,regret_simple,acqui.DeltaBt,acqui.rho_t)

        # Collect evaluation at xnext:
        y_new_obj   = function_obj(xnext)
        y_new_cons  = function_cons(xnext)

        # Update GP model:
        train_inputs_new        = torch.cat([gp_list[0].train_inputs[0], xnext])
        train_targets_obj_new   = torch.cat([gp_list[0].train_targets, y_new_obj.view(1)])
        train_targets_cons_new  = torch.cat([gp_list[1].train_targets, y_new_cons.view(1)])
        logvars["Xevals_array"] = train_inputs_new.detach().cpu().numpy()
        logvars["Yevals_obj_array"] = train_targets_obj_new.detach().cpu().numpy()
        logvars["Yevals_cons_array"] = train_targets_cons_new.detach().cpu().numpy()

        # Load GP model and fit hyperparameters:
        # NOTE: Re-instantiating the model is inefficient. However, it is not clear from the BoTorch documentation how to update the model
        # with new evaluations. Furthermore, this way is how it's done in the tutorials.
        del(gp_list)
        gp_obj = GPmodel(train_X=train_inputs_new, train_Y=train_targets_obj_new, options=cfg_node, which_type="obj")
        gp_obj.update_hyperparameters()
        gp_obj.display_hyperparameters()
        gp_cons = GPmodel(train_X=train_inputs_new, train_Y=train_targets_cons_new, options=cfg_node, which_type="cons")
        gp_cons.update_hyperparameters()
        gp_cons.display_hyperparameters()
        gp_list = [gp_obj,gp_cons]
        
        # Update the model in other classes:
        del(acqui); acqui = AcquiFun(model_list=gp_list, options=cfg_node)

    if cfg_node["thesis_plot_mode"] == True:
        path2save = "/Users/alonrot/MPI/WIP_papers/phd_thesis/oral_defense/wip/pics/code/XSF_expla/rho_t.pickle"
        import pickle
        rho_t_dict = dict(  rho_t_vec=rho_t_vec,
                            zk_t_vec=zk_t_vec,
                            DeltaBt_vec=DeltaBt_vec,
                            y_cons_vec=y_cons_vec,
                            readme="Saved from excursionsearch/xsearch/experiments/benchmarks/benchmarks_constrained.py")

        with open(path2save, "wb") as fid:
            pickle.dump(rho_t_dict, fid)

    node2write = convert_lists2arrays(logvars)
    node2write["n_rep"] = rep_nr
    node2write["ycm"] = f_min
    node2write["xcm"] = x_min
    node2write["params"] = cfg_node
    node2write["GPobj_pars"] = dict(X=logvars["Xevals_array"],Y=logvars["Yevals_obj_array"],
                                Xinit=logvars["Xinit"],Yinit=logvars["Yinit_obj"])
    node2write["GPcons_pars"] = dict(X=logvars["Xevals_array"],Y=logvars["Yevals_cons_array"],
                                Xinit=logvars["Xinit"],Yinit=logvars["Yinit_cons"])

    save_data(node2write=node2write,which_obj=cfg_node["which_objective"],which_acqui="XSF",rep_nr=rep_nr)

    return cfg_node["which_objective"]

if __name__ == "__main__":

    assert len(sys.argv) >= 2, "python run_benchmarks_unconstrained.py <BO algorithm> [<number_of_repetitions>]"

    which_algo = sys.argv[1]
    assert which_algo in ["XSF"], "which_algo = {'XSF'}"

    if len(sys.argv) == 3:
        Nrep = int(sys.argv[2])
        assert Nrep > 0
    else:
        Nrep = 1

    for rep_nr in range(Nrep):
        display_banner(which_algo,Nrep,rep_nr+1)
        ObjFun = run(rep_nr=rep_nr,which_algo=which_algo)

    # Convert data to a single file:
    if ObjFun != "simple1D":
        convert_from_cluster_data_to_single_file(which_obj=ObjFun,which_acqui=which_algo,Nrepetitions=Nrep)




