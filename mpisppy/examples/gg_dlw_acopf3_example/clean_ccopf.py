# This software is distributed under the 3-clause BSD License.
# JPW and DLW; July 2019; ccopf create scenario instances for line outages
# extended Fall 2019 by DLW
import egret
import egret.models.acopf as eac
from egret.data.model_data import ModelData
from egret.parsers.matpower_parser import create_ModelData
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.opt.ph
import mpisppy.examples.acopf3.ACtree as etree
import mpisppy.opt.aph
import mpisppy.examples.acopf3.rho_setter as rho_setter

import os
import sys
import copy
import scipy
import socket
import numpy as np
import datetime as dt
import mpi4py.MPI as mpi
from mpisppy.examples.gg_dlw_acopf3_example import utilities as util
from mpisppy.examples.gg_dlw_acopf3_example import rtsparser

import pyomo.environ as pyo

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size()

p = str(egret.__path__)
l = p.find("'")
r = p.find("'", l + 1)
egretrootpath = p[l + 1:r]
test_cases_path = {"IEEE": egretrootpath+"/thirdparty/pglib-opf-master/pglib_opf_case118_ieee.m",
              "RTS-GMLC": "RTS-GMLC"}

class threeStagesDispatch:

    def __init__(self, test_case="RTS-GMLC", relaxation=False, solvername="gurobi", verbose=False, a_line_fails_prob=0.2,
                 acstream=np.random.RandomState()):
        if relaxation is False and solvername is "gurobi":
            print("solver must be ipopt")
            solvername = "ipopt"
        self.test_case = test_case
        self.solver = pyo.SolverFactory(solvername)
        self.seed = 1234
        self.branching_factors = [2, 3]
        self.a_line_fails_prob = a_line_fails_prob
        self.repair_fct = util.FixFast
        self.number_of_stages = 3
        self.stage_duration_minutes = [60, 60, 60]
        self.verbose = verbose
        self.relaxation = relaxation
        self.acstream = acstream

        self.cb_data = dict()
        self.md_dict = None
        self.populate_cb_data()
        self.populate_md_dict()

        self.creator_options = {"cb_data": self.cb_data}
        self.scenario_names = ["Scenario_" + str(i) for i in range(1, len(self.cb_data["etree"].rootnode.ScenarioList)
                                                                   + 1)]
        self.ef = None

    def populate_cb_data(self):
        self.cb_data["solver"] = self.solver  # can be None
        self.cb_data["tee"] = False  # for inialization solve
        self.cb_data["epath"] = test_cases_path[self.test_case]
        self.cb_data["acstream"] = self.acstream

    def populate_md_dict(self):
        if self.test_case=="RTS-GMLC":
            begin_time = "2020-01-27 00:00:00"
            end_time = "2020-01-28 00:00:00"
            md_dict = rtsparser.create_ModelData(self.cb_data["epath"], begin_time, end_time, simulation="DAY_AHEAD",
                                                 t0_state=None)
            self.md_dict = util.from_piecewise_to_quadratic(md_dict)
        else:
            self.md_dict = create_ModelData(self.cb_data["epath"])

        if self.verbose:
            print("start data dump")
            print(list(self.md_dict.elements("generator")))
            for this_branch in self.md_dict.elements("branch"):
                print("TYPE=", type(this_branch))
                print("B=", this_branch)
                print("IN SERVICE=", this_branch[1]["in_service"])
            print("GENERATOR SET=", self.md_dict.attributes("generator"))
            print("end data dump")

        lines = list()
        for j, this_branch in enumerate(self.md_dict.elements("branch")):
            lines.append(str(j + 1))
            # lines.append(this_branch[0])

        self.cb_data["etree"] = etree.ACTree(self.number_of_stages,
                                        self.branching_factors,
                                        self.seed,
                                        self.acstream,
                                        self.a_line_fails_prob,
                                        self.stage_duration_minutes,
                                        self.repair_fct,
                                        list(self.md_dict.data["elements"]["branch"].keys()))


    def create_extensive_form(self):
        self.creator_options = {"cb_data": self.cb_data, "md_dict": self.md_dict, "first_time":0,
                                "relaxation": self.relaxation}
        self.scenario_names = ["Scenario_" + str(i) for i in range(1, len(self.cb_data["etree"].rootnode.ScenarioList)
                                                                   + 1)]
        self.ef = sputils.create_EF(self.scenario_names,
                                    util.pysp2_callback,
                               self.creator_options)
        return self.ef

if __name__ == '__main__':
    tsd = threeStagesDispatch(test_case="RTS-GMLC", verbose=False, relaxation=True)
    tsd.create_extensive_form()
    tsd_ieee = threeStagesDispatch(test_case="IEEE", verbose=True)