#include <iostream>
#include <random>

#include <ndt_map/ndt_map.h>
#include <ndt_map/ndt_cell.h>
#include <ndt_map/lazy_grid.h>
#include <ndt_map/pointcloud_utils.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;
typedef lslgeneric::NDTCell NDTCell;

pcl::PointCloud<pcl::PointXYZI> eigenToPCL(Eigen::MatrixXf& m)
{
  pcl::PointCloud<pcl::PointXYZI> xyz;
  xyz.points.reserve(m.rows());
  for (int i = 0; i < m.rows(); ++i)
  {
    pcl::PointXYZI p;
    p.x = m(i, 0);
    p.y = m(i, 1);
    p.z = m(i, 2);
    p.intensity = m(i, 3);
    xyz.points.push_back(p);
  }
  return xyz;
}

// m \in \mathbb{R}^{Nx3} where N is the number of
// data points
void loadPointCloud(lslgeneric::NDTMap& n, Eigen::MatrixXf& m)
{
  pcl::PointCloud<pcl::PointXYZI> xyz = eigenToPCL(m);
  n.loadPointCloud(xyz);
}

PYBIND11_MODULE(ndt_map, m)
{
  m.doc() = "NDT Map";

  py::class_<lslgeneric::LazyGrid>(m, "LazyGrid").def(py::init<double>());

  py::class_<lslgeneric::NDTCell>(m, "NDTCell")
      .def(py::init<>())
      .def("get_mean", &lslgeneric::NDTCell::getMean)
      .def("get_cov", &lslgeneric::NDTCell::getCov)
      .def("get_prec", &lslgeneric::NDTCell::getInverseCov)
      .def("get_evecs", &lslgeneric::NDTCell::getEvecs)
      .def("get_evals", &lslgeneric::NDTCell::getEvals)
      .def("get_intensity", &lslgeneric::NDTCell::getIntensity)
      .def("update_color_information",
           &lslgeneric::NDTCell::updateColorInformation)
      .def("get_support_size", [](NDTCell& c) { return c.points_.size(); });

  py::class_<lslgeneric::NDTMap>(m, "NDTMap")
      .def(py::init<lslgeneric::LazyGrid*>())
      .def("set_map_size", &lslgeneric::NDTMap::setMapSize)
      .def("get_grid_size", &lslgeneric::NDTMap::getGridSize)
      .def("compute_ndt_cells", &lslgeneric::NDTMap::computeNDTCells)
      .def("compute_ndt_cells_simple",
           &lslgeneric::NDTMap::computeNDTCellsSimple)
      .def("get_intensity_at_point",
           [](lslgeneric::NDTMap& m, float x, float y, float z) {
             lslgeneric::NDTCell* c;
             pcl::PointXYZ p(x, y, z);
             bool ret = m.getCellAtPoint(p, c);
             return std::pair<bool, float>(ret, c->getIntensity());
           })
      .def("get_all_cells",
           [](lslgeneric::NDTMap& m) { return m.getAllCells(); })
      .def("get_gaussians",
           [](lslgeneric::NDTMap& m) {
             std::vector<lslgeneric::NDTCell*> cells;
             cells = m.getAllCells();
             unsigned int num_components = cells.size();
             double uniform_weight = 1.0f / static_cast<float>(num_components);

             Eigen::VectorXd weights = Eigen::VectorXd::Zero(num_components);
             Eigen::MatrixXd means = Eigen::MatrixXd::Zero(num_components, 3);
             Eigen::MatrixXd covs = Eigen::MatrixXd::Zero(num_components, 9);
             for (unsigned int i = 0; i < num_components; i++)
             {
              weights(i) = uniform_weight;
              means.row(i) = cells[i]->getMean();
              Eigen::Matrix3d cov = cells[i]->getCov();
              covs.row(i) = Eigen::Map<Eigen::Vector<double, 9>>(cov.data(), cov.size());
             }

            return std::make_tuple(weights, means, covs);
           })
      .def("get_intensity_at_pcld",
           [](lslgeneric::NDTMap& m, const Eigen::MatrixXf& pcld) {
             Eigen::MatrixXf m_pcld = Eigen::MatrixXf::Zero(pcld.rows(), 4);
             for (unsigned int i = 0; i < pcld.rows(); i++)
             {
               lslgeneric::NDTCell* c;
               pcl::PointXYZ p(pcld(i, 0), pcld(i, 1), pcld(i, 2));
               bool ret = m.getCellForPoint(p, c);
               if (ret && c != NULL)
               {
                 m_pcld.row(i) << pcld(i, 0), pcld(i, 1), pcld(i, 2),
                     c->getIntensity();
               }
               else
               {
                 m_pcld.row(i) << pcld(i, 0), pcld(i, 1), pcld(i, 2), 0.0f;
               }
             }

             return m_pcld;
           })
      .def("load_pointcloud", &loadPointCloud);
}
