import networkx as nx
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.onehour_example_stations import *
import sys
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import alphashape
from scipy.stats import norm, beta
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.GMRF import GMRF
import numpy as np
import geojson, json, random
import pandas as pd
import numpy as np
from copy import deepcopy

locations, BBox = load_data()
xs, ys, station_locations_grid, station_locations_real, n_stations = set_up_grid_stations(0.05, BBox, locations)
# F = make_F(xs, ys, station_locations, len(station_locations), p=2.6)
# T_ = GMRF.tdata.forecast.iloc[:,locations.index].iloc[0].values
node_index = list(range(len(station_locations_grid)))

def generate_transformation_to_fit_Temp_grid(nodes):
    min_min_loc = np.min(np.array(station_locations_real)[:, 1]), np.min(np.array(station_locations_real)[:, 0])
    max_max_loc = np.max(np.array(station_locations_real)[:, 1]), np.max(np.array(station_locations_real)[:, 0])

    min_min_rts = np.min(nodes[:, 0]), np.min(nodes[:, 1])
    max_max_rts = np.max(nodes[:, 0]), np.max(nodes[:, 1])

    x_rts = max_max_rts[0] - min_min_rts[0]
    x_loc = max_max_loc[0] - min_min_loc[0]
    dx = x_loc / x_rts

    y_rts = max_max_rts[1] - min_min_rts[1]
    y_loc = max_max_loc[1] - min_min_loc[1]
    dy = y_loc / y_rts

    corner_left = [(min_min_loc[0], min_min_rts[0]), (max_max_loc[1], max_max_rts[1])]

    def from_rts_to_loc(node, corner_left):
        x = corner_left[0][0] + dx * (node[0] - corner_left[0][1])
        y = corner_left[1][0] + dy * (node[1] - corner_left[1][1])
        return [x, y]

    return from_rts_to_loc, corner_left

def get_everything():
    bus_df = pd.read_csv('RTS-GMLC/RTS_Data/SourceData/bus.csv')
    buses = list(bus_df.T.to_dict().values())

    bus_features = []

    bus_table = {}
    gen_count = {}

    for x in buses:
        bus_table[x['Bus ID']] = x
        gen_count['Bus ID'] = 0

        xy = x['lng'], x['lat']
        props = deepcopy(x)
        props.pop('lng')
        props.pop('lat')
        geom = geojson.Point(xy)
        f = geojson.Feature(geometry=geom, properties=props)
        bus_features.append(f)

    bus_collect = geojson.FeatureCollection(features=bus_features)
    nodes = np.array([p["geometry"]["coordinates"] for p in bus_collect["features"]])

    from_rts_to_loc, corner_left = generate_transformation_to_fit_Temp_grid(nodes)

    ##### Process branches #####
    branch_df = pd.read_csv('RTS-GMLC/RTS_Data/SourceData/branch.csv')
    branches = list(branch_df.T.to_dict().values())

    branch_features = []

    for x in branches:
        bf = bus_table[x['From Bus']]
        bt = bus_table[x['To Bus']]

        pf = bf['lng'], bf['lat']
        pt = bt['lng'], bt['lat']
        xy = pf, pt

        geom = geojson.LineString(xy)
        f = geojson.Feature(geometry=geom, properties=x)
        branch_features.append(f)

    branch_collect = geojson.FeatureCollection(features=branch_features)

    gen_df = pd.read_csv('RTS-GMLC/RTS_Data/SourceData/gen.csv')
    gens = list(gen_df.T.to_dict().values())

    gen_features = []
    gen_conn_features = []
    d = 0.1

    for x in gens:
        b = bus_table[x['Bus ID']]
        theta = random.uniform(-np.pi, np.pi)
        A = random.uniform(0,d)
        dx = A*np.cos(theta)
        dy = A*np.sin(theta)

        pg = b['lng'] + dx, b['lat'] + dy
        geom = geojson.Point(pg)
        f = geojson.Feature(geometry=geom, properties=x)
        gen_features.append(f)

        pb = b['lng'], b['lat']
        xy = pb, pg
        geom = geojson.LineString(xy)
        f = geojson.Feature(geometry=geom, properties=x)
        gen_conn_features.append(f)

    gen_collect = geojson.FeatureCollection(features=gen_features)
    gen_conn_collect = geojson.FeatureCollection(features=gen_conn_features)

    for i in range(len(gens)):
        # if gen_df.iloc[i]["Category"] == "Wind":
        lat, lng = bus_df[bus_df["Bus ID"] == gens[i]["Bus ID"]][["lat", "lng"]].values[0]
        gens[i]["location"] = from_rts_to_loc([lng, lat], corner_left)

    return gens



def plot_everything():
    import geojson, json, random
    with open('RTS-GMLC/RTS_Data/FormattedData/GIS/bus.geojson') as json_file:
        bus_collect = geojson.load(json_file)

    with open('RTS-GMLC/RTS_Data/FormattedData/GIS/branch.geojson') as json_file:
        branch_collect = geojson.load(json_file)

    nodes = np.array([p["geometry"]["coordinates"] for p in bus_collect["features"]])
    plt.scatter(nodes[:,0],nodes[:,1], marker = 'o', label = 'Simplified nodes')

    branches = [p["geometry"]["coordinates"] for p in branch_collect["features"]]
    m = len(branches)
    Tlines = [None] * (2*m)
    for l in range(m):
        Tlines[2*l] = (branches[l][0][0], branches[l][1][0])
        Tlines[2*l + 1] = (branches[l][0][1], branches[l][1][1])

    plt.plot(*Tlines, 'k')
    plt.show()


    #
    # if save:
    #     plt.savefig(title)
    # else:
    #     plt.show()


    min_min_loc = np.min(np.array(station_locations_real)[:,1]), np.min(np.array(station_locations_real)[:,0])
    max_max_loc = np.max(np.array(station_locations_real)[:,1]), np.max(np.array(station_locations_real)[:,0])

    min_min_rts = np.min(nodes[:,0]), np.min(nodes[:,1])
    max_max_rts = np.max(nodes[:,0]), np.max(nodes[:,1])

    x_rts = max_max_rts[0] - min_min_rts[0]
    x_loc = max_max_loc[0] - min_min_loc[0]
    dx = x_loc/x_rts

    y_rts = max_max_rts[1] - min_min_rts[1]
    y_loc = max_max_loc[1] - min_min_loc[1]
    dy = y_loc/y_rts

    corner_left = [(min_min_loc[0], min_min_rts[0]), (max_max_loc[1],  max_max_rts[1])]

    def from_rts_to_loc(node, corner_left):
        x = corner_left[0][0] + dx*(node[0]-corner_left[0][1])
        y = corner_left[1][0] + dy*(node[1]-corner_left[1][1])
        return [x, y]

    node =  bus_collect["features"][0]["geometry"]["coordinates"]
    nodes = np.array([from_rts_to_loc(p["geometry"]["coordinates"], corner_left) for p in bus_collect["features"]])
    plt.scatter(nodes[:,0],nodes[:,1], marker = 'o', label = 'Simplified nodes')

    branches = [[from_rts_to_loc(p["geometry"]["coordinates"][0], corner_left),
                 from_rts_to_loc(p["geometry"]["coordinates"][1], corner_left)] for p in branch_collect["features"]]
    m = len(branches)
    Tlines = [None] * (2*m)
    for l in range(m):
        Tlines[2*l] = (branches[l][0][0], branches[l][1][0])
        Tlines[2*l + 1] = (branches[l][0][1], branches[l][1][1])

    plt.plot(*Tlines, 'k')
    plt.show()


    """
    Real example
    """
    gen_locations = []
    for i in range(len(gen_df)):
        if gen_df.iloc[i]["Category"] == "Wind":
            lat, lng = bus_df[bus_df["Bus ID"]==gen_df.iloc[i]["Bus ID"]][["lat", "lng"]].values[0]
            gen_locations.append(from_rts_to_loc([lng, lat], corner_left))
    gen_locations = np.array(gen_locations)

    z_f = F.dot(T_)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.set_title("RTS GMLC + Temperature Grid")
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])

    h = ax.contourf(xs, ys, z_f, alpha=0.7, cmap=cm.plasma)

    ax.scatter(gen_locations[:,0], gen_locations[:,1], marker="x", color="black", s=1000)
    ax.plot(xs, ys, 'k-', lw=0.5, alpha=0.2)
    ax.plot(xs.T, ys.T, 'k-', lw=0.5, alpha=0.2)
    ax.scatter(locations.longitude, locations.latitude, zorder=1, alpha=0.6, c='b', s=10)
    nodes = np.array([from_rts_to_loc(p["geometry"]["coordinates"], corner_left) for p in bus_collect["features"]])
    ax.scatter(nodes[:,0],nodes[:,1], marker = 'o', label = 'Simplified nodes')
    branches = [[from_rts_to_loc(p["geometry"]["coordinates"][0], corner_left),
                 from_rts_to_loc(p["geometry"]["coordinates"][1], corner_left)] for p in branch_collect["features"]]
    m = len(branches)
    Tlines = [None] * (2*m)
    for l in range(m):
        Tlines[2*l] = (branches[l][0][0], branches[l][1][0])
        Tlines[2*l + 1] = (branches[l][0][1], branches[l][1][1])

    ax.plot(*Tlines, 'k')
    for i in node_index:
        ax.annotate(str(i), (station_locations_grid[i][0], station_locations_grid[i][1]))
    fig.colorbar(h, ax=ax)
    plt.show()



    from pykrige.ok import OrdinaryKriging
    import numpy as np

    # Make this example reproducible:
    np.random.seed(89239413)

    # Generate random data following a uniform spatial distribution
    # of nodes and a uniform distribution of values in the interval
    # [2.0, 5.5]:
    N = 7
    lon = 360.0 * np.random.random(N)
    lat = 180.0 / np.pi * np.arcsin(2 * np.random.random(N) - 1)
    z = 3.5 * np.random.rand(N) + 2.0

    # Generate a regular grid with 60° longitude and 30° latitude steps:
    grid_lon = np.linspace(0.0, 360.0, 7)
    grid_lat = np.linspace(-90.0, 90.0, 7)

    # Create ordinary kriging object:
    OK = OrdinaryKriging(
        lon,
        lat,
        z,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
    )

    # Execute on grid:
    z1, ss1 = OK.execute("grid", grid_lon, grid_lat)

    # Create ordinary kriging object ignoring curvature:
    OK = OrdinaryKriging(
        lon, lat, z, variogram_model="linear", verbose=False, enable_plotting=False
    )

    # Execute on grid:
    z2, ss2 = OK.execute("grid", grid_lon, grid_lat)

    plt.contourf(grid_lon, grid_lon, z2, alpha=0.7, cmap=cm.plasma)
    plt.show()

    """
    Kriging
    """

    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging


    # Make this example reproducible:
    np.random.seed(89239413)

    N = 25
    lon = np.array(station_locations_grid)[:,0]
    lat = np.array(station_locations_grid)[:,1]
    z = T_

    grid_lon = xs[0]
    grid_lat = ys[:,0]

    # Create ordinary kriging object ignoring curvature:
    OK = UniversalKriging(
        lon, lat, z, variogram_model='exponential',
        drift_terms=['regional_linear'], verbose=True, enable_plotting=False)
    OK = OrdinaryKriging(
        lon, lat, z, variogram_model='linear', verbose=True, enable_plotting=False)


    # Execute on grid:
    z2, ss2 = OK.execute("grid", grid_lon, grid_lat)




    """
    Mean temperature one line
    """
    lines = dict([[i, [from_rts_to_loc(branch_collect["features"][i]["geometry"]["coordinates"][0], corner_left),
                 from_rts_to_loc(branch_collect["features"][i]["geometry"]["coordinates"][1], corner_left)]] for i in range(m)])


    def T_line(line):
        x_array = np.linspace(line[0][0], line[1][0], 100)
        slope = (line[1][1]-line[0][1])/(line[1][0]-line[0][0])
        y_array = line[0][1] + slope*(x_array-line[0][0])

        t_line, ss2 = OK.execute("points", x_array, y_array)
        return np.mean(t_line)


    t_lines = dict([[i, T_line(lines[i])] for i in range(m)])

    plt.contourf(xs, ys, z2, alpha=0.7, cmap=cm.plasma)
    plt.show()


    # data = np.array([[0.3, 1.2, 0.47],
    #                  [1.9, 0.6, 0.56],
    #                  [1.1, 3.2, 0.74],
    #                  [3.3, 4.4, 1.47],
    #                  [4.7, 3.8, 1.74]])
    #
    # gridx = np.arange(0.0, 5.5, 0.5)
    # gridy = np.arange(0.0, 5.5, 0.5)
    #
    # # Create the ordinary kriging object. Required inputs are the X-coordinates of
    # # the data points, the Y-coordinates of the data points, and the Z-values of the
    # # data points. If no variogram model is specified, defaults to a linear variogram
    # # model. If no variogram model parameters are specified, then the code automatically
    # # calculates the parameters by fitting the variogram model to the binned
    # # experimental semivariogram. The verbose kwarg controls code talk-back, and
    # # the enable_plotting kwarg controls the display of the semivariogram.
    # OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
    #                      verbose=False, enable_plotting=False)
    #
    # # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
    # # grid of points, on a masked rectangular grid of points, or with arbitrary points.
    # # (See OrdinaryKriging.__doc__ for more information.)
    # z, ss = OK.execute('grid', gridx, gridy)


    # import numpy as np
    # from pykrige.rk import Krige
    # from pykrige.compat import GridSearchCV
    #
    #
    # # 2D Kring param opt
    #
    # param_dict = {
    #     "method": ["ordinary", "universal"],
    #     "variogram_model": ["linear", "power", "gaussian", "spherical"],
    # }
    #
    # estimator = GridSearchCV(Krige(), param_dict, verbose=True)
    #
    # # # dummy data
    # # X = np.random.randint(0, 400, size=(100, 2)).astype(float)
    # # y = 5 * np.random.rand(100)
    #
    # # run the gridsearch
    # estimator.fit(X=np.array(station_locations_grid), y=T_)
    #
    #
    # if hasattr(estimator, "best_score_"):
    #     print("best_score R² = {:.3f}".format(estimator.best_score_))
    #     print("best_params = ", estimator.best_params_)
    #
    # print("\nCV results::")
    # if hasattr(estimator, "cv_results_"):
    #     for key in [
    #         "mean_test_score",
    #         "mean_train_score",
    #         "param_method",
    #         "param_variogram_model",
    #     ]:
    #         print(" - {} : {}".format(key, estimator.cv_results_[key]))


"""
Load generators
"""
gens = get_everything()