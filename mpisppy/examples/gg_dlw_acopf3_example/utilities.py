import numpy as np
import scipy
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
import egret.models.acopf as eac
import egret.models.ac_relaxations as eac_relax
import copy
import mpisppy.scenario_tree as scenario_tree


def piecewise(x, pcost):
    bpts = [pcost["values"][i][0] for i in range(len(pcost['values']))]
    ydata = [pcost["values"][i][1] for i in range(len(pcost['values']))]
    i = 0
    while i < len(bpts) and bpts[i] <= x:
        i += 1
    slope = (ydata[i] - ydata[i - 1]) / (bpts[i] - bpts[i - 1])
    return slope * (x - bpts[i - 1]) + ydata[i - 1]


def from_piecewise_to_quadratic(md_dict):
    for n in md_dict.data["elements"]["generator"].keys():
        if "p_cost" in md_dict.data["elements"]["generator"][n]:
            p_cost = md_dict.data["elements"]["generator"][n]["p_cost"]
            if p_cost["cost_curve_type"] == "piecewise":
                if len(p_cost["values"]) == 1:
                    coeffs = [p_cost["values"][0][1], 0., 0.]
                else:
                    p_cost = md_dict.data["elements"]["generator"][n]["p_cost"]
                    p_w_function = lambda x: piecewise(x, md_dict.data["elements"]["generator"][n]["p_cost"])
                    pmin, pmax = md_dict.data["elements"]["generator"][n]['p_min'], md_dict.data["elements"]["generator"][n]['p_max']
                    if pmax > pmin:
                        dp = (pmax-pmin)/1000
                        ps = np.linspace(pmin+dp, pmax-dp, 100)
                        function_values = [p_w_function(p) for p in ps]
                        coeffs = np.polyfit(ps, function_values, 2)
                        for i in range(len(coeffs)):
                            if coeffs[i] < 1e-2:
                                coeffs[i] = 0
                    else:
                        coeffs = [0., 0., 0.]
                md_dict.data["elements"]["generator"][n]["p_cost"] = {'data_type': 'cost_curve',
                                                                      'cost_curve_type': 'polynomial',
                                                                      'values': {0: coeffs[-1],  1: coeffs[1], 2: coeffs[0]}}
                # print(md_dict.data["elements"]["generator"][n]["p_cost"])
        else:
            md_dict.data["elements"]["generator"][n]["p_cost"] = {'data_type': 'cost_curve',
                                                                      'cost_curve_type': 'polynomial',
                                                                      'values': {0: 0.,  1: 0., 2: 0.}}
        print(n, md_dict.data["elements"]["generator"][n]["p_cost"])

    return md_dict

def change_coramin_objective(model):
    stored_obj = model.aux_objectives  # coramin is setting a new objective named aux_objectives
    model.del_component("aux_objectives")
    model.obj = stored_obj[1]
    return model

def from_timeseries_to_values(first_stage_md_dict, this_time):
    name_of_loads = list(first_stage_md_dict.data["elements"]["load"].keys())
    for n in name_of_loads:
        one_period_pload = first_stage_md_dict.data["elements"]["load"][n]["p_load"]["values"][this_time]
        one_period_qload = first_stage_md_dict.data["elements"]["load"][n]["q_load"]["values"][this_time]
        first_stage_md_dict.data["elements"]["load"][n]["p_load"] = one_period_pload
        first_stage_md_dict.data["elements"]["load"][n]["q_load"] = one_period_qload

    gen_attrs = first_stage_md_dict.attributes(element_type='generator')
    p_mins_by_gen = gen_attrs["p_min"]
    p_maxs_by_gen = gen_attrs["p_max"]
    for name_gen in gen_attrs["names"]:
        if type(p_mins_by_gen[name_gen]) is not float:
            first_stage_md_dict.data["elements"]["generator"][name_gen]["p_min"] = \
            gen_attrs["p_min"][name_gen]["values"][this_time]
        if type(p_maxs_by_gen[name_gen]) is not float:
            first_stage_md_dict.data["elements"]["generator"][name_gen]["p_max"] = \
            gen_attrs["p_max"][name_gen]["values"][this_time]

    return first_stage_md_dict


def FixFast(minutes):
    return True

def FixNever(minutes):
    return False

def FixGaussian(minutes, acstream, mu, sigma):
    """
    Return True if the line has been repaired.
    Args:
        minutes (float) : how long has the line been down
        mu, sigma (float): repair time is N(mu, sigma)
    """
    # spell it out...
    Z = (minutes-mu)/sigma
    u = acstream.rand()
    retval = u < scipy.norm.cdf(Z)
    return retval


def pysp2_callback(scenario_name, node_names=None, cb_data=None, md_dict=None, relaxation=False, first_time=0):
    """
    mpisppy signature for scenario creation. Basically, just call the PySP1 fct.
    Then find a starting solution for the scenario if solver option is not None.
    Note that stage numbers are one-based.

    Args:
        scenario_name (str): put the scenario number on the end
        node_names (int): not used
        cb_data: (dict) "etree", "solver", "epath", "tee"

    Returns:
        scenario (pyo.ConcreteModel): the scenario instance

    Attaches:
        _enodes (ACtree nodes): a list of the ACtree tree nodes
        _egret_md (egret tuple with dict as [1]) egret model data

    """
    # pull the number off the end of the scenario name
    scen_num = sputils.extract_num(scenario_name)

    etree = cb_data["etree"]
    solver = cb_data["solver"]
    acstream = cb_data["acstream"]

    # seed each scenario every time to avoid troubles
    acstream.seed(etree.seed + scen_num)
    """
    inst = pysp_instance_creation_callback(scenario_tree_model = etree,
                                           scenario_name = scenario_name)
    """

    def lines_up_and_down(stage_md_dict, enode):
        # local routine to configure the lines in stage_md_dict for the scenario
        LinesDown = []
        for f in enode.FailedLines:
            LinesDown.append(f[0])
        for this_branch in stage_md_dict.elements("branch"):
            if this_branch[0] in enode.LinesUp:
                this_branch[1]["in_service"] = True
            elif this_branch[0] in LinesDown:
                this_branch[1]["in_service"] = False
            else:
                print("enode.LinesUp=", enode.LinesUp)
                print("enode.FailedLines=", enode.FailedLines)
                raise RuntimeError("Branch (line) {} neither up nor down in scenario {}". \
                                   format(this_branch[0], scenario_name))

    # pull the number off the end of the scenario name
    scen_num = sputils.extract_num(scenario_name)
    # print ("debug scen_num=",scen_num)

    numstages = etree.NumStages
    enodes = etree.Nodes_for_Scenario(scen_num)
    full_scenario_model = pyo.ConcreteModel()
    full_scenario_model.stage_models = dict()

    # the exact acopf model is hard-wired here:
    if relaxation:
        acopf_model = eac_relax.create_soc_relaxation
    else:
        acopf_model = eac.create_riv_acopf_model

    first_stage_md_dict = copy.deepcopy(md_dict)
    saved_md_dict = copy.deepcopy(md_dict)
    first_stage_md_dict = from_timeseries_to_values(first_stage_md_dict, first_time)

    # the following creates the first stage model
    full_scenario_model.stage_models[1], model_dict = acopf_model(first_stage_md_dict, include_feasibility_slack=True)
    full_scenario_model.stage_models[1] = change_coramin_objective(full_scenario_model.stage_models[1])

    """
    Addition
    """
    generator_set = model_dict.attributes("generator")
    generator_names = generator_set["names"]

    if hasattr(full_scenario_model.stage_models[1], "obj"):
        full_scenario_model.stage_models[1].obj.deactivate()
    setattr(full_scenario_model,
            "stage_models_" + str(1),
            full_scenario_model.stage_models[1])

    for stage in range(2, numstages + 1):
        print ("stage={}".format(stage))
        stage_md_dict = copy.deepcopy(saved_md_dict)
        stage_md_dict = from_timeseries_to_values(stage_md_dict, first_time+stage-1)
        print ("debug: processing node {}".format(enodes[stage-1].Name))
        lines_up_and_down(stage_md_dict, enodes[stage - 1])

        full_scenario_model.stage_models[stage], model_dict = acopf_model(stage_md_dict, include_feasibility_slack=True)
        full_scenario_model.stage_models[stage] = change_coramin_objective(full_scenario_model.stage_models[stage])
        if hasattr(full_scenario_model.stage_models[stage], "obj"):
            full_scenario_model.stage_models[stage].obj.deactivate()
        setattr(full_scenario_model,
                "stage_models_" + str(stage),
                full_scenario_model.stage_models[stage])

    def aggregate_ramping_rule(m):
        """
        We are adding ramping to the obj instead of a constraint for now
        because we may not have ramp limit data.
        """
        retval = 0
        for stage in range(1, numstages):
            retval += sum((m.stage_models[stage + 1].pg[this_gen]\
                           - m.stage_models[stage].pg[this_gen]) ** 2\
                          for this_gen in generator_names)
        return retval

    full_scenario_model.ramping = pyo.Expression(rule=aggregate_ramping_rule)

    full_scenario_model.objective = pyo.Objective(expr= \
                                                      1000000.0 * full_scenario_model.ramping + \
                                                      sum(full_scenario_model.stage_models[stage].obj.expr \
                                                          for stage in range(1, numstages + 1)))

    inst = full_scenario_model
    # end code from PySP1

    node_list = list()

    parent_name = None
    for sm1, enode in enumerate(etree.Nodes_for_Scenario(scen_num)):
        stage = sm1 + 1
        if stage < etree.NumStages:
            node_list.append(scenario_tree.ScenarioNode(
                name=enode.Name,
                cond_prob=enode.CondProb,
                stage=stage,
                cost_expression=inst.stage_models[stage].obj,
                scen_name_list=enode.ScenarioList,
                nonant_list=[inst.stage_models[stage].pg,
                             inst.stage_models[stage].qg],
                scen_model=inst, parent_name=parent_name))
            parent_name = enode.Name

    inst._PySPnode_list = node_list
    # Optionally assign probability to PySP_prob
    inst.PySP_prob = 1 / etree.numscens
    # solve it so subsequent code will have a good start
    if solver is not None:
        solver.solve(inst, tee=True)

    # attachments
    inst._enodes = enodes
    inst._egret_md = first_stage_md_dict

    return inst