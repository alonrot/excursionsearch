Description
=========
This python package `excursionsearch` contains two complementary algorithms: Excursion Search (XS) and Failures-aware excursion search (XSF).

Excursion Search (XS) is a Bayesian optimization (BO) based on _excursion sets theory_. The proposed strategy suggests queries where the unknown objective, modeled with a Gaussian process (GP), is expected up- or down-cross the current estimate of the global optimum. Intuitively, this implies querying in areas near a possible maximum. In practice, the queries are collected where the GP process crosses the level of the estimated optimum with large norm. Contrary to existing BO method, our algorithm explicitly uses the gradient of the GP for decision-making. Our results show superior performance with respect to state-of-the-art methods (e.g., EI, PI, UCB, MES).

Failures-aware excursion search (XSF) uses the aforementioned XS method to solve constrained BO problems when a budget of failures is given. This type of setting is particularly recurrent in the industry, where constraint violation is undesirable, but not catastrophic. In this case, failures can be seen as a rich source of information about what should not be done. Hence, using a limited number of failures could play in our favor, thus speeding up the learning process. The proposed method attempts to balances the decision-making between (i) safely exploring encountered safe areas, and (ii) searching outside the safe areas at the risk of failing, when safe areas contain no further information.

Both algorithms are explained in our submission

	Alonso Marco, Alexander von Rohr, Dominik Baumann, José Miguel Hernández-Lobato, and Sebastian Trimpe, 
	"Excursion Search for Constrained Bayesian Optimization under a Limited Budget of Failures", 
	2020.



Requirements
============

The two proposed algorithms, `Excursion Search (XS)` and `Failures-aware Excursion Search (XSF)`, run in Python >= 3.6, and are developed under [BoTorch](https://botorch.org/).

> If your python installation does not meet the minimum requirement, we recommend creating a virtual environment with the required python version. For example, [Anaconda](https://www.anaconda.com/distribution/) allows this, and does not interfere with your system-wide Python installation underneath. 

> NOTE: We recommend opening this README.md file in an online Markdown editor, e.g., [StackEdit](https://stackedit.io/app#), for better readability.

[BoTorch](https://botorch.org/) is a flexible framework for developing new Bayesian optimization algorithms. It builts on [Pytorch](https://pytorch.org/) and uses [scipy Python optimizers](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html) for GP model fitting and acquisition function optimization. Currently, BoTorch does not support numerical optimization under the presence of non-linear constraints, which is needed to optimize `XSF` (cf. eq. (13) in the paper). To overcome this, we use the Python API of [nlopt](https://nlopt.readthedocs.io/en/latest/).


Installation 
============

Without Anaconda
----------------

1. Make sure your python version meets the required one. For this, open a terminal and type
```bash
python --version
```
2. Install the following dependencies
```bash
pip install botorch>=0.2.1
pip install matplotlib>=3.1.3
pip install nlopt>=2.6.1
pip install pyyaml>=5.3
```
3. Extract the contents of `code_excursionsearch.zip`, provided in the supplementary material, to your desired path <path/to/excursionsearch>
4. Navigate to the package folder and install it
```bash
cd <path/to/excursionsearch>
pip install -e .
```

With Anaconda
----------------

Following these instructions, a new conda environment named `xsearch_env` will be created.
It is assumed that Anaconda is already installed. If not, you can download and install from [here](https://www.anaconda.com/distribution/).
1. Extract the contents of `code_excursionsearch.zip`, provided in the supplementary material, to <path/to/excursionsearch>
2. Navigate to the package folder, create the conda environment from a generated environment file, and activate it
```bash 
cd <path/to/excursionsearch>
conda env create -f condaenv.yaml
conda activate xsearch_env
```

> NOTE: If the above fails, try creating a Python 3.6 environment and installing manually the dependencies, i.e.,
```bash 
conda create -n xsearch_env python=3.6
conda activate xsearch_env
conda install botorch -c pytorch -c gpytorch
conda install matplotlib
conda install pyyaml
pip install nlopt
cd <path/to/excursionsearch>
```

3. Install the package excursionsearch
```bash
pip install -e .
```

Fix a bug in BoTorch
--------------------

We detected a minor incompatibility between BoTorch and [PyTorch](https://pytorch.org/). When computing the random restarts for optimizing acquisition functions, BoTorch attempts to use `torch.multinomial`, a specific PyTorch library, by passing a 2D tensor, when actually it only admits 1D tensors (see [here](https://pytorch.org/docs/stable/torch.html?highlight=multinomial#torch.multinomial)). This error can be solved by modifying the file `botorch/optim/initializers.py` inside the BoTorch package. To fix it, you need to first locate your python installation. For this, open a terminal and type
```bash
which python
```
You should see a path of the form <root/path>/bin/python. 

Now, open the aforementioned file with an editor. If using `vim`:
```bash
vim <root/path>/lib/pythonX.X/site-packages/botorch/optim/initializers.py
```
Replace the line 280:
```python
idcs = torch.multinomial(weights, n)
```
with
```python
idcs = torch.multinomial(weights.view(-1), n)
```

Running `Excursion Search (XS)` (cf. Sec. 3 in the paper)
=========================================================

```bash
cd xsearch/experiments/benchmarks/
python run_bench.py XS
```
By default, the **toy mode** is activated, which shows a 1D optimization experiment. A window should pop up, showing the GP model and the `XS` acquisition function. In the terminal, you should see verbose information about the progress of the algorithm, e.g., regret evolution and hyperparameters updates after fitting GP models. The algorihtm runs for 8 iterations. The initial GP is conditioned on 7 evaluations, the same as those shown in Figure 1 in the paper. 
<!-- ![XS_toy_mode](/pics/XS_toy_mode.png) -->
For an example of the plots generated by `XS`, please open the file `pics/XS_toy_mode.png`.


Running `Failures-aware Excursion Search (XSF)` (cf. Sec. 4 in the paper)
=========================================================================

```bash
cd xsearch/experiments/benchmarks/
python run_bench.py XSF
```

By default, the **toy mode** is activated, which shows a 1D *constrained* optimization experiment. You shold see a window popping up, which plots the GP that models the objective function, the GP that models the constraint, and the `XSF` acquisition function. In addition, the safe/unsafe areas where the constraint is satisfied/violated are shown with light colors. The green triangle indicates the minimum of the posterior mean s.t. the probabilistic constraint is satisfied (cf. Algorithm 1 in Appendix C returns the minimizer of this quantity).
<!-- ![XSF_toy_mode](/pics/XSF_toy_mode.png) -->
For an example of the plots generated by `XSF`, please open the file `pics/XSF_toy_mode.png`.


Reproducing results
===================

To reproduce the results from Table 1 in the paper, for `XS` and `XSF` algorithms, the **toy mode** must be deactivated and either the Michalewicz (10D) or the Hartman (6D) function must be selected. This can be done manually by editing the configuration file
`xsearch/experiments/benchmarks/conf_bench.yaml`. For example, for running experiments with the Hartman (6D) function
```yaml
which_objective: "hart6D"
# which_objective: "micha10D"
toy_mode: False
```
To run multiple experiments (sequentially), run the scripts adding an extra input argument
```bash
python run_bench.py XSF <Nrep>
python run_bench.py XS <Nrep>
```
where `<Nrep>` is the desired number of repetitions (1 by default if omitted).
Some new paths will be automatically created to store the results as the experiments are running. For example, for the Hartman (6D) experiments with `XS`, the new paths will be
`xsearch/experiments/benchmarks/hart6D/XS_results`
`xsearch/experiments/benchmarks/hart6D/XS_results/cluster_data`

After the experiments are finished, an automatic data conversion will parse the collected data to a single file and stored in a folder named after the current date. In the terminal, you should see a message like
```bash
[xsearch.utils.parse_data_collection] Saving in ./hart6D/XS_results/20200210133219/data.yaml
```
which indicates that the data has been stored in `xsearch/experiments/benchmarks/hart6D/XS_results/20200210133219`.

Plotting results
----------------
To plot the stored results, run the following script
```bash
python plot_results.py XSF <ObjFun> <which_acqui> [<nr_exp>]
```
where `<ObjFun>` is the true function (`micha10D` or `hart6D`), `<which_acqui>` refers to the used algorithm (`XS` or `XSF`) and `[<nr_exp>]` is the experiment name you want to load the data from (e.g., `20200210133219`). This last argument is optional, and if omitted, the most recent experiment will be loaded. For example, after running the aforementioned experiments with the Hartman (6D) function for 10 repetitions, you could plot them by running
```bash
python plot_results.py hart6D XS
```

General comments
================

 * All the hyperparameters mentioned in paper can be found in `xsearch/experiments/benchmarks/conf_bench.yaml`, and modified. The current values correspond to those mentioned in the paper.
 * The first time any of the above algorithms are run, they can take a few seconds to start.

Known issues for macOS users
============================
 * If any of the aforementioned plots do not automatically pop up, try uncommenting line 3 in the files `xsearch/utils/plotting_collection.py` and `xsearch/experiments/benchmarks/plot_results.py`
```python
matplotlib.use('TkAgg') # Solves a no-plotting issue for macOS users
```
 * If you encounter [this](https://stackoverflow.com/questions/56989086/unable-to-install-nlopt-python-package) issue while installing `nlopt`, your gcc compiler might not be working correctly (this can happen after a recent macOS upgrade). You might need to accept the [xcode license agreements](https://stackoverflow.com/questions/22844522/os-x-10-9-2-checking-whether-the-c-compiler-works-no).

 * Support for using GPU is at the moment not fully implemented.
 
 * If you encounter problems while installing PyTorch, check [here](https://pytorch.org/get-started/locally/).


