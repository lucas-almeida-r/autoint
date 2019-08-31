
#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <Sacado.hpp>

using namespace dealii;

class Solver
{
public:
  Solver(const int poly_degree, const unsigned int refine_global, const unsigned int quad_degree);
  ~Solver();
  void run();
  
private:
  void solve();

  const unsigned int refine_global, quad_degree;

  const double pressure = 1000, c11 = 1e+5, c22 = 5e+4, c23 = 5.5e+3,
               eps = 0.1, radius = 1, c12 = 5e+4, gama = (c22 + c23 + c12)/c11;
  double delta = 1;

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> F_delta, E_h, P_h, DE_h, DP_h;

  Triangulation<1> triangulation;
  DoFHandler<1>    dof_handler;

  FE_Q<1> fe;

  std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> solution;
  std::vector<double> grad_F;
  std::vector<std::vector<double>> hess_F;
};