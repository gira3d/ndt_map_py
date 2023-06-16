import numpy as np
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.mixture import GaussianMixture

from sogmm_py.utils import np_to_o3d, matrix_to_tensor
from ndt_map import NDTMap, LazyGrid

# Adapted from the following sources:
# 1. https://stackoverflow.com/a/7820701
# 2. https://github.com/rislab/publisher_utils/blob/6270b35057a184e304b022463881d209c1731f40/include/publisher_utils/GMM3Publisher.h


class NDTViz:
    sigma_ = 1

    def __init__(self, desired_sigma):
        self.sigma_ = desired_sigma

    # Coefficients in (x/a)**2 + (y/b)**2 + (z/c)**2 = 1
    # a,b,c lie on the surface of the ellipsoid
    def eigen_values_to_coefs(self, evals):

        # Eigen values represent variance along the x,y,z directions
        # standard deviation is equal to sqrt of eigen value
        a = np.sqrt(evals[0]) * self.sigma_
        b = np.sqrt(evals[1]) * self.sigma_
        c = np.sqrt(evals[2]) * self.sigma_

        return np.array([(1/a)**2, (1/b)**2, (1/c)**2])

    def plot(self, ndt_cell, ax):
        mean = ndt_cell.get_mean()

        coefs = self.eigen_values_to_coefs(ndt_cell.get_evals())

        # Radii corresponding to the coefficients:
        rx, ry, rz = 1/np.sqrt(coefs)

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v))
        y = ry * np.outer(np.sin(u), np.sin(v))
        z = rz * np.outer(np.ones_like(u), np.cos(v))

        # Rotate using the eigen vectors as Rotation matrix
        R = ndt_cell.get_evecs()
        if np.linalg.det(R) < 0:
            R *= -1

        tx = np.zeros([np.shape(x)[0], np.shape(x)[1]])
        ty = np.zeros([np.shape(y)[0], np.shape(y)[1]])
        tz = np.zeros([np.shape(z)[0], np.shape(z)[1]])
        for i in range(0, np.shape(x)[0]):
            for j in range(0, np.shape(x)[1]):
                t = np.dot(R, np.array([[x[i][j]], [y[i][j]], [z[i][j]]]))
                tx[i][j] = t[0] + mean[0]
                ty[i][j] = t[1] + mean[1]
                tz[i][j] = t[2] + mean[2]

        ax.plot_surface(tx, ty, tz,  rstride=4, cstride=4, color='b')


def main():
    l = LazyGrid(0.02)
    n = NDTMap(l)
    data = np.load('copier.npz')
    pcld = np.array(data["arr_0"])
    n_samples = np.shape(pcld)[0]

    n.load_pointcloud(pcld)
    n.compute_ndt_cells_simple()

    weights, means, covs = n.get_gaussians()
    weights /= np.sum(weights)
    n_components = np.shape(weights)[0]

    g = GaussianMixture(n_components=n_components, covariance_type='full')
    g.weights_ = weights
    g.means_ = means
    g.covariances_ = matrix_to_tensor(covs, 3)

    recon_pcld, _ = g.sample(n_samples)
    # o3d_m_pcld = np_to_o3d(recon_pcld)
    # o3d.visualization.draw_geometries([o3d_m_pcld])

    m_pcld = n.get_intensity_at_pcld(recon_pcld[:, 0:3])
    o3d_m_pcld = np_to_o3d(m_pcld)
    o3d.visualization.draw_geometries([o3d_m_pcld])


if __name__ == "__main__":
    main()
