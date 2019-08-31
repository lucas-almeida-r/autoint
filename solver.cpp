
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>
#include <cmath>

#include <Sacado.hpp>

#include "solver.h"


Solver::Solver(const int poly_degree, const unsigned int refine_global, const unsigned int quad_degree): 
  refine_global(refine_global),
  quad_degree(quad_degree),
  dof_handler(triangulation),
  fe(poly_degree)
{}

Solver::~Solver()
{}


void Solver::solve ()
{
  const bool analytic_diff = true;
  const QGauss<1>  quadrature_formula(quad_degree);
  FEValues<1> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> sg, sg_prime; // scalar_product(s,g) e (s,g')

  F_delta = 0;
  E_h = 0;
  P_h = 0;
  DE_h = 0;
  DP_h = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      // se tiver que achar derivadas ja usando os dofs locais precisa desse codigo comentado
      /* std::vector<Sacado::Fad::DFad<double>> local_etas(dofs_per_cell);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        local_etas[i] = solution[local_dof_indices[i]];
        local_etas[i].diff(i, dofs_per_cell);
      } */

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        sg = 0;
        sg_prime = 0;
        double rho = fe_values.quadrature_point(q)[0]; // coordenada global
      
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double phi_i = fe_values.shape_value(i, q);
          const double phi_i_prime = fe_values.shape_grad(i, q)[0]; // acessa o seu unico elemento
          sg += solution[local_dof_indices[i]] * phi_i;
          sg_prime += solution[local_dof_indices[i]] * phi_i_prime;
        }

        E_h += (pow(rho*sg_prime, 2) + 2*gama*pow(sg,2)) * fe_values.JxW(q);
        P_h += pow(rho, 2) / ( (1+sg_prime) * pow(1+sg/rho, 2) - eps ) * fe_values.JxW(q);

        // Calculo derivadas analiticas de eta_1 (para o eta_n falta adicionar termos ao DE_h)
        if(analytic_diff)
        {
          int i_aux = (local_dof_indices[0] == 1) ? 0 : (local_dof_indices[1] == 1) ? 1 : 999;
          if(i_aux != 999)
          {
            DE_h += (2*pow(rho,2)*sg_prime*fe_values.shape_grad(i_aux, q)[0] + 
                  4*gama*sg*fe_values.shape_value(i_aux, q)) * fe_values.JxW(q);
            DP_h += -pow(rho,2) * 
                    ( fe_values.shape_grad(i_aux, q)[0] * pow(1+sg/rho, 2) +
                      (1+sg_prime)*2*(1+sg/rho)*fe_values.shape_value(i_aux, q)/rho ) / 
                    pow((1+sg_prime)*pow(1+sg/rho,2) - eps, 2) * 
                    fe_values.JxW(q);
          }
        }
        
      }
      // TALVEZ DIVIDIR em um "forward phase" e em um "update phase" e nao deixar tudo em um "solve"
      // ai o forward calcularia F_h e os gradientes, e depois rodaria o update para atualizar o s
    }
    E_h = c12 * pow(solution.back(), 2) * radius + 
          pressure * solution.back() * pow(radius, 2) +
          c11/2 * E_h;
    F_delta = E_h + P_h / delta;

    if(analytic_diff)
      DE_h = c11/2*DE_h;

    // montagem do gradiente
    for(unsigned int i = 0; i < solution.size(); ++i)
      grad_F_delta[i] = F_delta.dx(i).val();

    // print do gradiente
    for(unsigned int i = 0; i < solution.size(); ++i)
      std::cout << grad_F_delta[i] << std::endl;

    // alguns prints para conferir valores
    std::cout << "E_h: " << E_h << "\nP_h: " << P_h << "\nF_delta: " << F_delta 
              << "\nDE_h: " << DE_h <<  "\nDP_h: " << DP_h << "\n";


}


void Solver::run ()
{
  // temos 8 ciclos de refinamento, no primeiro ciclo a gente cria a mesh e refinamos 1 vez
  // nos outros ciclos, chamamos a funcao refine_grid 
  for (unsigned int cycle=0; cycle<1; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;
      if (cycle == 0)
        {
          GridGenerator::hyper_cube(triangulation, 0, radius, /*colorize*/ true);
          triangulation.refine_global(refine_global);
        }
      else
        Assert(false, ExcNotImplemented()); //refine_grid ();

      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

      dof_handler.distribute_dofs(fe); // garantir numeracao do rho=0 ate o rho=radius
      //solution.resize(dof_handler.n_dofs(), 0.001); // lembrar que primeiro dof é sempre 0
      solution = {0.001,0.003,0.004};
      grad_F_delta.resize(dof_handler.n_dofs(), 0);
      for (unsigned int i = 0; i < solution.size(); ++i) // diz que solution sao variaveis independentes
        {
          solution[i].diff(i, solution.size());
          solution[i].val().diff(i, solution.size());
        }
        
      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;
    
    solve();
    //std::cout << gama << std::endl;
    }
}