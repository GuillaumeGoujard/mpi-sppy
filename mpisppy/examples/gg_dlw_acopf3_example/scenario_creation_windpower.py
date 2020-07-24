import numpy as np
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.GMRF import GMRF
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.FilterClassNew import temp_median, no_filter
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.GMRF import to_minnesota_time
import datetime
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.onehour_example_stations import *
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RTS_GMLC_loc import RTS_MISOgrid
from pykrige.ok import OrdinaryKriging


def windturbine_shutoff(temperature, min_temp=0):
    if temperature > min_temp:
        return 1
    else:
        return 0


def simulate_scenarios(n):
    gmrf = GMRF(number_of_days_fit=1, filter=temp_median())

    start_time = to_minnesota_time(datetime.datetime(2019, 12, 25, 0, 0))
    end_time = to_minnesota_time(datetime.datetime(2019, 12, 25, 12, 0))
    T_scenarios = gmrf.estimate_and_simulate(start_time, end_time, n_scenarios=n, estimate=True)
    return T_scenarios


# z = T_scenarios[0,0,:]

def create_wind_gens_list():
    wind_gens = []
    loc_wind_gens = []
    for i in range(len(RTS_MISOgrid.gens)):
        if RTS_MISOgrid.gens[i]["Category"] == "Wind":
            wind_gens.append(RTS_MISOgrid.gens[i])
            wind_gens[-1]["temperature"] = []
            wind_gens[-1]["status"] = []
            loc_wind_gens.append(RTS_MISOgrid.gens[i]["location"])
    loc_wind_gens = np.array(loc_wind_gens)
    return wind_gens, loc_wind_gens


def update_status_one_s_one_t(z, wind_gens, loc_wind_gens):
    np.random.seed(89239413)

    N = 25
    lon = np.array(RTS_MISOgrid.station_locations_grid)[:,0]
    lat = np.array(RTS_MISOgrid.station_locations_grid)[:,1]

    grid_lon = xs[0]
    grid_lat = ys[:,0]

    OK = OrdinaryKriging(
        lon, lat, z, variogram_model='linear', verbose=True, enable_plotting=False)

    # Execute on grid:
    z2, ss2 = OK.execute("grid", grid_lon, grid_lat)

    loc_wind_gens = np.array(loc_wind_gens)
    t_gen, ss2 = OK.execute("points", loc_wind_gens[:,0], loc_wind_gens[:,1])
    for i in range(len(t_gen)):
        wind_gens[i]["temperature"].append(t_gen[i])
        wind_gens[i]["status"].append(windturbine_shutoff(t_gen[i]))

    return wind_gens


def getwindturbineScenarios(n):
    T_scenarios = simulate_scenarios(n)
    wind_gens, loc_wind_gens = create_wind_gens_list()
    wind_t_scenarios = []
    for i in range(T_scenarios.shape[0]):
        for t in range(T_scenarios.shape[1]):
            wind_gens = update_status_one_s_one_t(T_scenarios[i][t], wind_gens, loc_wind_gens)
        wind_t_scenarios.append(wind_gens)
    return wind_t_scenarios