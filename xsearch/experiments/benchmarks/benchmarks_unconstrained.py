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
from xsearch.acquisitions import Xsearch
from xsearch.objectives import Hartmann6D, Michalewicz10D, Simple1DSequentialGP
from xsearch.models.gpmodel import GPmodel
from botorch.utils.sampling import draw_sobol_samples
from xsearch.utils.parsing import convert_lists2arrays, save_data, display_banner, get_logger
import logging
from xsearch.utils.plotting_collection import plotting_tool_uncons
import yaml
from xsearch.utils.parse_data_collection import convert_from_cluster_data_to_single_file
logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def initialize_logging_variables():
    logvars = dict( mean_bg_list=[],
                    x_bg_list=[],
                    x_next_list=[],
                    alpha_next_list=[],
                    Xevals=[],
                    Yevals=[],
                    regret_simple_list=[],
                    Xinit=0,
                    Yinit=0)
    return logvars

def append_logging_variables(logvars,eta,x_eta,x_next,alpha_next,regret_simple):
    logvars["mean_bg_list"].append(eta.view(1).detach().cpu().numpy())
    logvars["x_bg_list"].append(x_eta.view(x_eta.shape[1]).detach().cpu().numpy())
    logvars["x_next_list"].append(x_next.view(x_eta.shape[1]).detach().cpu().numpy())
    logvars["alpha_next_list"].append(alpha_next.view(1).detach().cpu().numpy())
    logvars["regret_simple_list"].append(regret_simple.view(1).detach().cpu().numpy())
    return logvars

def get_initial_evaluations(which_objective):

    assert which_objective in ["hart6D", "micha10D", "simple1D"], "Objective function <which_objective> must be {'hart6D','micha10D','simple1D'}"

    if which_objective == "hart6D":
        train_x = torch.Tensor([[0.32124528, 0.00573107, 0.07254258, 0.90988337, 0.00164314, 0.41116992]]) # Randomly computed
        train_y = torch.Tensor([0.4980]) # train_y = function_obj(train_x)
    
    if which_objective == "micha10D":
        train_x = torch.Tensor([[0.90999505,0.54672184,0.26503819,0.58101023,0.60157458,0.94950461,0.06963194,0.59811682,0.88415646,0.23745016]]) # Randomly computed
        train_y = torch.Tensor([0.3966]) # train_y = function_obj(train_x)

    if which_objective == "simple1D":
        train_x = torch.Tensor([[0.93452506],[0.18872502],[0.89790337],[0.95841797],[0.82335255],[0.45000000],[0.50000000]]) # 7 evaluations
        train_y = torch.Tensor([-0.4532849,-0.66614552,-0.92803395,0.08880341,-0.27683621,1.000000,1.500000]) # 7 evaluations

    return train_x, train_y

def get_function_obj(which_objective,gp=None):

    assert which_objective in ["hart6D", "micha10D", "simple1D"], "Objective function <which_objective> must be {'hart6D','micha10D','simple1D'}"

    if which_objective == "hart6D":
        func_obj = Hartmann6D()
        dim = 6
    if which_objective == "micha10D":
        func_obj = Michalewicz10D()
        dim = 10
    if which_objective == "simple1D":
        func_obj = Simple1DSequentialGP(gp)
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
    if which_algo == "XS":
        AcquiFun = Xsearch

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
    else:
        cfg_node["plotting"] = False

    # Random seed for numpy and torch:
    np.random.seed(rep_nr)
    torch.manual_seed(rep_nr)

    # Load initial evaluations:
    train_x, train_y = get_initial_evaluations(cfg_node["which_objective"])

    # Load GP model and fit hyperparameters:
    gp = GPmodel(train_X=train_x, train_Y=train_y, options=cfg_node)
    gp.update_hyperparameters()
    gp.display_hyperparameters()
    
    # Initialize true objective function:
    function_obj, dim, x_min, f_min = get_function_obj(cfg_node["which_objective"],gp)

    # Initialize acquisition function:
    AcquiFun = get_acquisition_function(which_algo)
    acqui = AcquiFun(model=gp, options=cfg_node)

    logvars = initialize_logging_variables()
    logvars["Xinit"] = train_x.detach().cpu().numpy()
    logvars["Yinit"] = train_y.detach().cpu().numpy()

    # Plotting:
    if cfg_node["plotting"]:
        axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gp,acqui,axes_GPobj=None,axes_acqui=None,axes_fmin=None)

    # average over multiple trials
    for trial in range(cfg_node["NBOiters"]):
        
        msg_bo_iters = " <<< BO Iteration {0:d} / {1:d} >>>".format(trial+1,cfg_node["NBOiters"])
        print("\n")
        logger.info("="*len(msg_bo_iters))
        logger.info("{0:s}".format(msg_bo_iters))
        logger.info("="*len(msg_bo_iters))

        # Get next point:
        xnext, alpha_next = acqui.get_next_point()

        # Compute simple regret:
        regret_simple = acqui.get_simple_regret(fmin_true=f_min)
        logger.info("Regret: {0:2.5f}".format(regret_simple.item()))
        
        # Logging:
        if cfg_node["plotting"]:
            axes_GPobj, axes_acqui, axes_fmin = plotting_tool_uncons(gp,acqui,axes_GPobj,axes_acqui,axes_fmin,xnext=xnext,alpha_next=alpha_next)
        append_logging_variables(logvars,acqui.eta,acqui.x_eta,xnext,alpha_next,regret_simple)

        # Collect evaluation at xnext:
        y_new = function_obj(xnext)

        # Update GP model:
        train_inputs_new = torch.cat([gp.train_inputs[0], xnext])
        train_targets_new = torch.cat([gp.train_targets, y_new.view(1)])
        logvars["Xevals_array"] = train_inputs_new.detach().cpu().numpy()
        logvars["Yevals_array"] = train_targets_new.detach().cpu().numpy()
        
        # Load GP model and fit hyperparameters:
        # NOTE: Re-instantiating the model is inefficient. However, it is not clear from the BoTorch documentation how to update the model
        # with new evaluations. Furthermore, this way is how it's done in the tutorials.
        del(gp); gp = GPmodel(train_X=train_inputs_new, train_Y=train_targets_new, options=cfg_node)
        gp.update_hyperparameters()
        gp.display_hyperparameters()
        if dim == 1: function_obj = Simple1DSequentialGP(gp)
        
        # Update the model in other classes:
        del(acqui); acqui = AcquiFun(model=gp, options=cfg_node)

    node2write = convert_lists2arrays(logvars)
    node2write["n_rep"] = rep_nr
    node2write["ycm"] = f_min
    node2write["xcm"] = x_min
    node2write["params"] = cfg_node
    node2write["GPobj_pars"] = dict(X=logvars["Xevals_array"],Y=logvars["Yevals_array"],
                                Xinit=logvars["Xinit"],Yinit=logvars["Yinit"])

    save_data(node2write=node2write,which_obj=cfg_node["which_objective"],which_acqui="XS",rep_nr=rep_nr)

    return cfg_node["which_objective"]

if __name__ == "__main__":

    assert len(sys.argv) >= 2, "python run_benchmarks_unconstrained.py <BO algorithm> [<number_of_repetitions>]"

    which_algo = sys.argv[1]
    assert which_algo in ["XS"], "which_algo = {'XS'}"

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




