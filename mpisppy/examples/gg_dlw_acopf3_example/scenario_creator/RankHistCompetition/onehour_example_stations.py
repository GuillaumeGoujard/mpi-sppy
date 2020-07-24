import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import pandas as pd
import threading
from PIL import Image
import matplotlib.cm as cm
import matplotlib as mpl
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def load_data():
    # gridrows = 5
    # gridcolumns = 8
    # tdata = TemperatureClass.TemperatureClass('mpisppy.examples.gg_dlw_acopf3_example/data100/xfer/results100.csv', gridrows, gridcolumns, input_start_dt=None)
    locations = pd.read_csv(dir_path + '/data100/locations.csv', sep=',')
    locations = locations[locations["longitude"] < -88]
    locations = locations + np.random.normal(0, 0.001, locations.shape)

    # BBox = ((locations.longitude.min(),   locations.longitude.max(),
    #          locations.latitude.min(), locations.latitude.max()))
    BBox = (-94.500, -88.000, 39.500, 44.000)
    return locations, BBox


def nearest_nodes(station_locations):
    coords = [station_locations[k] for k in range(len(station_locations))]
    tree = spatial.KDTree(coords)
    neighbour_list = [list(tree.query(station_locations[i], k=8)[1]) for i in range(len(station_locations))]
    return tree

def plot_map():
    fname = 'map.png'
    image = Image.open(fname).convert("L")
    arr = np.asarray(image)
    fig, ax = plt.subplots(figsize=(8,4), dpi=100)
    ax.scatter(locations.longitude, locations.latitude, zorder=1, alpha= 0.2, c='b', s=10)
    ax.set_title('Plotting Spatial Data on Minesota')
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.imshow(arr, zorder=0, extent=BBox, cmap='gray', aspect='equal', interpolation='nearest')
    plt.show()

def set_up_grid_stations(dx, BBox, locations):
    # nx, ny = nx_, nx_
    x = np.arange(BBox[0], BBox[1], dx)
    y = np.arange(BBox[2], BBox[3], dx)
    xs, ys = np.meshgrid(x, y)

    station_locations_grid = [(0,0)]*len(locations)
    station_locations = [(0,0)]*len(locations)
    for i in range(len(locations)):
        y_, x_ = locations.iloc[i].values
        station_locations_grid[i] = (x[np.argmin(abs(x_-x))], y[np.argmin(abs(y_-y))])
        station_locations[i] = locations.iloc[i].values
    n_stations = len(station_locations)
    return xs, ys, station_locations_grid, station_locations, n_stations

def Temperatures_a_f(tdata, i):
    T_a = tdata.observed.iloc[i].values
    T_f = tdata.forecast.iloc[i].values
    return T_a, T_f


# print("loading data")
locations, BBox = load_data()
# print("loading nodes")
xs, ys, station_locations, _, n_stations = set_up_grid_stations(0.2, BBox, locations)
# print("loading temperature")
# T_a, T_f = Temperatures_a_f(tdata, 0)
# k = 8


def d(x, y, i=0, p=2.6):
    a, b = np.array([x,y]), np.array(station_locations[i])
    return np.power(np.linalg.norm(a-b), p)


def IDW(x, y, T):
    denom, num = 0, 0
    for i in range(n_stations):
        if (x, y) == station_locations[i]:
            return T[i]
    for i in range(len(station_locations)): #list(node_tree.query(station_locations[i], k=k))[1]:
        num += T[i]*(1/d(x, y, i=i))
        denom += 1/d(x, y, i=i)
    return num/denom


def plot_with_IDW1(xs, ys, T, title="observed", z=None, vmin=None, vmax=None, levels=None):
    # fname = 'map.png'
    # image = Image.open(fname).convert("L")
    # arr = np.asarray(image)
    if z is None:
        z_a = np.zeros(xs.shape)
        for i in range(xs.shape[0]):
            for j in range(xs.shape[1]):
                z_a[i,j] = IDW(xs[i,j], ys[i,j], T)
    else:
        z_a = z

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.set_title(title)
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    # ax.imshow(arr, zorder=0, extent=BBox, cmap='gray', aspect='equal', interpolation='nearest')
    ax.plot(xs, ys, 'k-', lw=0.5, alpha=0.2)
    ax.plot(xs.T, ys.T, 'k-', lw=0.5, alpha=0.2)
    if vmin is not None and levels is None:
        h = ax.contourf(xs, ys, z_a, alpha=0.7, cmap=cm.jet, levels=20, vmin=vmin, vmax=vmax)
    elif levels is not None:
        h = ax.contourf(xs, ys, z_a, alpha=0.7, cmap=cm.jet, levels=levels, vmin=vmin, vmax=vmax)
    else:
        h = ax.contourf(xs, ys, z_a, alpha=0.7, cmap=cm.jet, levels=20)
    ax.scatter(locations.longitude, locations.latitude,  zorder=1, alpha=0.6, c='b', s=10)
    # plt.colorbar()
    fig.colorbar(h, ax=ax)
    plt.show()
    return h.levels


def plot_with_IDW2(xs, ys, T_a, T_f):
    # fname = 'map.png'
    # image = Image.open(fname).convert("L")
    # arr = np.asarray(image)
    z_a = np.zeros(xs.shape)
    z_f = np.zeros(xs.shape)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            z_a[i,j] = IDW(xs[i,j], ys[i,j], T_a)
            z_f[i, j] = IDW(xs[i, j], ys[i, j], T_f)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    axs = [ax1, ax2]
    ax1.set_title("actuals")
    ax2.set_title("forecasts")
    for ax in axs:
        ax.set_xlim(BBox[0], BBox[1])
        ax.set_ylim(BBox[2], BBox[3])
        # ax.imshow(arr, zorder=0, extent=BBox, cmap='gray', aspect='equal', interpolation='nearest')
        ax.plot(xs, ys, 'k-', lw=0.5, alpha=0.2)
        ax.plot(xs.T, ys.T, 'k-', lw=0.5, alpha=0.2)
        ax.scatter(locations.longitude, locations.latitude, zorder=1, alpha=0.6, c='b', s=10)
    vmin, vmax = np.min([np.min(z_f), np.min(z_a)]),  np.max([np.max(z_f), np.max(z_a)])
    ha = ax1.contourf(xs, ys, z_a, cmap=cm.jet, alpha=0.7, vmin=vmin, vmax=vmax, levels=None)
    hf = ax2.contourf(xs, ys, z_f, cmap=cm.jet, alpha=0.7,  vmin=vmin, vmax=vmax,levels=ha.levels)
    cax, kw = mpl.colorbar.make_axes(axs)
    plt.colorbar(ha, cax=cax, **kw)
    plt.savefig("test.png")
    plt.show()

def make_f(x, y, station_locations, n_stations, p=2.6):
    denom = 0
    f = np.zeros(n_stations)
    for i in range(n_stations):
        if (x, y) == station_locations[i]:
            f[i] = 1
            return f
    for i in range(n_stations):
        denom += 1 / d(x, y, i=i, p=p)
    f = [(1/d(x, y, i=i, p=p))/denom for i in range(n_stations)]
    return np.array(f)


def make_F(xs, ys, station_locations, n_stations, p=2.6):
    F = np.zeros([xs.shape[0], xs.shape[1], n_stations])
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            F[i, j] = make_f(xs[i, j], ys[i, j], station_locations, n_stations, p=p)
    return F

def plot_with_vectorial_operation(F, xs, ys):
    T_hat = F.dot(T_a)
    h = plt.contourf(xs, ys, T_hat)
    plt.plot(xs, ys, 'k-', lw=0.5, alpha=0.5)
    plt.plot(xs.T, ys.T, 'k-', lw=0.5, alpha=0.5)
    plt.colorbar()
    plt.title("Temperature field")
    plt.show()



def gradient(F, xs, ys):
    dx = xs[0][1]-xs[0][0]
    dy = ys[1][0]-ys[0][0]
    G = np.zeros((xs.shape[0], xs.shape[1], 2, n_stations))
    for s in range(n_stations):
        g_y_0, g_x_0 = np.gradient(F[:, :, s], dy, dx)
        G[:, :, 0, s] = g_x_0
        G[:, :, 1, s] = g_y_0
    return G


def grad_mag_grad(F, xs, ys):
    dx = xs[0][1]-xs[0][0]
    dy = ys[1][0]-ys[0][0]
    G = gradient(F, xs, ys)
    G_ = np.zeros((xs.shape[0], xs.shape[1], 2))
    G_a_g = np.array([[np.linalg.norm(G[i, j, :,:], ord=1) for j in range(G.shape[1])] for i in range(G.shape[0])])
    g_y_0, g_x_0 = np.gradient(G_a_g[:, :], dy, dx)
    G_[:, :, 0] = g_x_0
    G_[:, :, 1] = g_y_0
    return G_

def Hessian(G, xs, ys):
    dx = xs[0][1] - xs[0][0]
    dy = ys[1][0] - ys[0][0]
    H = np.zeros((xs.shape[0], xs.shape[1], 2, 2, n_stations))
    for s in range(n_stations):
        gxy, gxx = np.gradient(G[:, :, 0, s], dx)
        gyy, gyx = np.gradient(G[:, :, 1, s], dy)
        H[:, :, 0, 0, s] = gxx
        H[:, :, 0, 1, s] = gxy
        H[:, :, 1, 1, s] = gyy
        H[:, :, 1, 0, s] = gyx
    return H


def magnitude_gradient(G, T_a, xs, ys):
    dx = xs[0][1] - xs[0][0]
    dy = ys[1][0] - ys[0][0]
    l = G.dot(T_a)
    L = np.array([[np.linalg.norm(l[i, j], ord=1) for j in range(l.shape[1])] for i in range(l.shape[0])])
    G_ = np.zeros((xs.shape[0], xs.shape[1], 2))
    g_y_0, g_x_0 = np.gradient(L[:, :], dy, dx)
    G_[:, :, 0] = g_x_0
    G_[:, :, 1] = g_y_0
    return G_


def magnitude_gradient_2(x, y):
    dx, dy = 0.01, 0.01
    S = np.zeros((2, n_stations))
    for s in range(n_stations):
        Z_x = np.array([make_f(x - dx, y)[s], make_f(x, y)[s], make_f(x + dx, y)[s]])
        Z_y = np.array([make_f(x, y - dy)[s], make_f(x, y)[s], make_f(x, y + dy)[s]])
        S[0, s] = np.gradient(Z_x, dx)[1]
        S[1, s] = np.gradient(Z_y, dy)[1]
    return np.linalg.norm(S,1)



if __name__ == '__main__':
    # tdata, locations, BBox = load_data()
    # xs, ys, station_locations, n_stations = set_up_grid_stations(10, BBox, locations)
    # nearest_nodes = nearest_nodes(station_locations)
    T_a, T_f = Temperatures_a_f(tdata, 0)

    plot_with_IDW1(T_a, title="observed")
    plot_with_IDW1(T_f, title="forecasts")
    plot_with_IDW1(T_f-T_a, title="errors")
    # plot_with_IDW2(xs, ys, T_a, T_f)

    F = make_F(xs, ys, station_locations, n_stations)
    T_hat = F.dot(T_a) #scalar field
    plot_with_IDW1(T_a, title="observed", z=T_hat)
    T_hat_fore = F.dot(T_f)  # scalar field
    plot_with_IDW1(T_f, title="forecasted", z=T_hat_fore)

    G = gradient(F, xs, ys)
    H = Hessian(G, xs, ys)
    G_m_g = grad_mag_grad(F, xs, ys)

    l = G.dot(T_a)
    mg = magnitude_gradient(G, T_a, xs, ys)
    real_front_z = np.zeros((xs.shape[0], xs.shape[1]))
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            real_front_z[i][j] = mg[i, j].dot(l[i,j])
    # real_front_z[real_front_z > -10] = 0
    plot_with_IDW1(T_f, title="real locator", z=real_front_z)

    fname = 'map.png'
    image = Image.open(fname).convert("L")
    arr = np.asarray(image)
    z_a = real_front_z
    real_front_z[abs(real_front_z)>50]=0

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.set_title("real locator")
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.imshow(arr, zorder=0, extent=BBox, cmap='gray', aspect='equal', interpolation='nearest')
    ax.plot(xs, ys, 'k-', lw=0.5, alpha=0.2)
    ax.plot(xs.T, ys.T, 'k-', lw=0.5, alpha=0.2)
    h = ax.contourf(xs, ys, z_a, alpha=0.7, cmap=cm.jet, levels=100)
    ax.scatter(locations.longitude, locations.latitude, zorder=1, alpha=0.6, c='b', s=10)
    # plt.colorbar()
    fig.colorbar(h, ax=ax)
    plt.show()

    # h = G.T.dot(G_m_g)

    z_loc = np.zeros((xs.shape[0], xs.shape[1]))
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            z_loc[i][j] = G_m_g[i, j].dot(G[i,j].dot(T_a).T)
    z_loc[z_loc > 0] = 0
    plot_with_IDW1(T_f, title="locator", z=z_loc)

    z_G = np.zeros((xs.shape[0], xs.shape[1]))
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            z_G[i, j] = np.linalg.norm(G[i, j].dot(T_a))
    plot_with_IDW1(T_a, title="norm of gradient", z=z_G)

    z_H = np.zeros((xs.shape[0], xs.shape[1]))
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            z_H[i, j] = np.linalg.norm(H[i, j].dot(T_a))
    plot_with_IDW1(T_a, title="norm of Hessian", z=z_H)


    fname = 'map.png'
    image = Image.open(fname).convert("L")
    arr = np.asarray(image)
    z_a = z_H

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.set_title("Hessian")
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    ax.imshow(arr, zorder=0, extent=BBox, cmap='gray', aspect='equal', interpolation='nearest')
    ax.plot(xs, ys, 'k-', lw=0.5, alpha=0.2)
    ax.plot(xs.T, ys.T, 'k-', lw=0.5, alpha=0.2)
    h = ax.contourf(xs, ys, z_a, alpha=0.7, cmap=cm.jet, levels=100, vmax=50)
    ax.scatter(locations.longitude, locations.latitude, zorder=1, alpha=0.6, c='b', s=10)
    # plt.colorbar()
    fig.colorbar(h, ax=ax)
    plt.show()

    from scipy.ndimage import gaussian_filter
    z2 = gaussian_filter(T_hat, sigma=1)
    plot_with_IDW1(T_a, title="observed", z=z2)
