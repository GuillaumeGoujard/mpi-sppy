# general example driver for farmer with cylinders and an l-shape hub
# NOTE: as of June 2020, it does not use the vanilla cylinders

import mpisppy.examples.farmer.farmer as farmer

# Make it all go
from mpisppy.utils.sputils import spin_the_wheel
from mpisppy.examples import baseparsers
from mpisppy.examples import vanilla
from mpisppy.cylinders.hub import LShapedHub
from mpisppy.opt.lshaped import LShapedMethod


def _parse_args():
    parser = baseparsers.make_parser(num_scens_reqd=True)
    parser = baseparsers.two_sided_args(parser)
    parser = baseparsers.fwph_args(parser)
    parser.add_argument("--crops-mult",
                        help="There will be 3x this many crops (default 1)",
                        dest="crops_mult",
                        type=int,
                        default=1)                
    parser.add_argument("--threads",
                        help="Value for threads option (e.g. 1; default None)",
                        dest="threads",
                        type=int,
                        default=None)
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    num_scen = args.num_scens
    crops_mult = args.crops_mult
    with_fwph = args.with_fwph

    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = [f"scen{sn}" for sn in range(num_scen)]
    cb_data={"use_integer": False, "CropsMult": crops_mult}
    scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    # Things needed for vanilla cylinders
    beans = (args, scenario_creator, scenario_denouement, all_scenario_names)

    # Options for the L-shaped method at the hub
    # Bounds only valid for 3 scenarios, I think? Need to ask Chris
    spo = None if args.threads is None else {"threads": args.threads}
    options = {
        "master_solver": args.solver_name,
        "sp_solver": args.solver_name,
        "sp_solver_options" : spo,
        "valid_eta_lb": {i: -432000 for i in all_scenario_names},
        "max_iter": 10,
        "verbose": False,
   }
    
    # L-shaped hub
    hub_dict = {
        "hub_class": LShapedHub,
        "hub_kwargs": {
            "options": {
                "rel_gap": args.rel_gap,
                "abs_gap": args.abs_gap,
            },
        },
        "opt_class": LShapedMethod,
        "opt_kwargs": { # Args passed to LShapedMethod __init__
            "options": options,
            "all_scenario_names": all_scenario_names,
            "scenario_creator": scenario_creator,
            "cb_data": cb_data,
        },
    }

    # FWPH spoke
    if with_fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, cb_data=cb_data)

    list_of_spoke_dict = list()
    if with_fwph:
        list_of_spoke_dict.append(fw_spoke)

    spin_the_wheel(hub_dict, list_of_spoke_dict)


if __name__ == "__main__":
    main()