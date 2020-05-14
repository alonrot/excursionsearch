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
from xsearch.utils.parse_data_collection import convert_from_cluster_data_to_single_file
from xsearch.utils.parsing import display_banner
import sys

if __name__ == "__main__":

    assert len(sys.argv) >= 2, "python run_benchmarks_unconstrained.py <BO algorithm> [<number_of_repetitions>]"

    which_algo = sys.argv[1]
    assert which_algo in ["XS","XSF"], "which_algo = {'XS',XSF}"

    if len(sys.argv) == 3:
        Nrep = int(sys.argv[2])
        assert Nrep > 0
    else:
        Nrep = 1

    if which_algo == "XS":
    	from benchmarks_unconstrained import run
    elif which_algo == "XSF":
    	from benchmarks_constrained import run

    for rep_nr in range(Nrep):
        display_banner(which_algo,Nrep,rep_nr+1)
        ObjFun = run(rep_nr=rep_nr,which_algo=which_algo)

    # Convert data to a single file:
    if ObjFun != "simple1D":
        convert_from_cluster_data_to_single_file(which_obj=ObjFun,which_acqui=which_algo,Nrepetitions=Nrep)


