
#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>

#include <Sacado.hpp>

using namespace dealii;

class MySolver
{
public:
  MySolver(const int poly_degree, const unsigned int refine_global, const unsigned int quad_degree);
  ~MySolver();
  void run();
  
private:
  void solve();
  void compute_F_grad_hess();
  void compute_dk();
  void compute_alpha();
  void compute_alpha_derivs(double alpha, double &dF_dAlpha, double &d2F_dAlpha2);
  void create_480_cells();
  void write_output_file();

  const unsigned int refine_global, quad_degree;

  const double pressure = 1000, c11 = 1e+5, c22 = 5e+4, c23 = 5.5e+3,
               eps = 0.1, radius = 1, c12 = 5e+4, gama = (c22 + c23 - c12)/c11;
  double delta = 0.001, delta_max = 1e+8;
  unsigned int n_dofs;
  double alpha_k;
  bool verbose = false;
  const unsigned int iter_limit_alpha = 100; // numero maximo de iteracoes para alpha
  const unsigned int iter_limit_sk = 100; // numero maximo de iteracoes para s_k
  const double alpha_tol = 0.0001; // criterio de parada da busca pelo alpha
  const double solution_tol = 0.0001; // criterio de parada da busca pelo s_k
  std::ofstream output_file;

  Triangulation<1> triangulation;
  DoFHandler<1>    dof_handler;

  FE_Q<1> fe;

  //std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> solution;
  std::vector<double> solution, prev_solution;
  std::vector<double> grad_F, dk;
  std::vector<std::vector<double>> hess_F;

  // comp_grad, comp_dk, comp_alpha, update+parada
  std::vector<double> timing{0.,0.,0.,0.};
  // output data
  std::vector<std::vector<double>> output_data;
};