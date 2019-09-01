
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

void Solver::compute_F_grad_hess()
{
  const bool analytic_diff = true;

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> F_delta, E_h, P_h, DE_h, DP_h, DDE_h;
  std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> dofs(n_dofs, 0);

  // inicializacao de F_delta, E_h, ... , dofs
  F_delta = 0;
  E_h = 0;
  P_h = 0;
  DE_h = 0;
  DP_h = 0;
  for (unsigned int i = 0; i < n_dofs; ++i) // diz que solution sao variaveis independentes
  {
    dofs[i] = solution[i];
    dofs[i].diff(i, n_dofs);
    dofs[i].val().diff(i, n_dofs);
  }

  const QGauss<1>  quadrature_formula(quad_degree);
  FEValues<1> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);
  
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> sg, sg_prime; // scalar_product(s,g) e (s,g')

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    // se tiver que achar derivadas ja usando os dofs locais precisa desse codigo comentado
    /* std::vector<Sacado::Fad::DFad<double>> local_etas(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      local_etas[i] = dofs[local_dof_indices[i]];
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
        sg += dofs[local_dof_indices[i]] * phi_i;
        sg_prime += dofs[local_dof_indices[i]] * phi_i_prime;
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
                    4*gama*sg*fe_values.shape_value(i_aux, q)) * 
                  fe_values.JxW(q);
          DP_h += -pow(rho,2) * 
                  ( fe_values.shape_grad(i_aux, q)[0] * pow(1+sg/rho, 2) +
                    (1+sg_prime)*2*(1+sg/rho)*fe_values.shape_value(i_aux, q)/rho ) / 
                  pow((1+sg_prime)*pow(1+sg/rho,2) - eps, 2) * 
                  fe_values.JxW(q);
          DDE_h += (2*pow(rho, 2)*pow(fe_values.shape_grad(i_aux, q)[0], 2) + 
                    4*gama*pow(fe_values.shape_value(i_aux, q),2)) *
                    fe_values.JxW(q);
        }
      }
    }
  } // end for cells

  E_h = c12 * pow(dofs.back(), 2) * radius + 
        pressure * dofs.back() * pow(radius, 2) +
        c11/2 * E_h;
  F_delta = E_h + P_h / delta;

  if(analytic_diff)
  {
    DE_h = c11/2 * DE_h;
    DDE_h = c11/2 * DDE_h;
  }
    
  // montagem do gradiente e da hessiana
  for(unsigned int i = 0; i < n_dofs; ++i)
  {
    grad_F[i] = F_delta.dx(i).val();
    for(unsigned int j =0; j < n_dofs; ++j)
      hess_F[i][j] = F_delta.dx(i).dx(j);
  }

  // alguns prints para conferir valores
  std::cout << "\nE_h: " << E_h << "\nP_h: " << P_h << "\nF_delta: " << F_delta 
            << "\nDE_h: " << DE_h <<  "\nDP_h: " << DP_h << "\nDDE_h: " << DDE_h << "\n";
}

void Solver::compute_dk()
{
  // cria o Tensor que vai conter o gradiente e a hessiana
  Vector<double> gradT(grad_F.begin(), grad_F.end()), dkT;
  dkT.reinit(n_dofs);
  FullMatrix<double> hessT(n_dofs, n_dofs), inv_hessT(n_dofs, n_dofs);

  for (unsigned int i = 0; i < n_dofs; ++i)
  {
    gradT(i) = grad_F[i];
    for (unsigned int j = 0; j < n_dofs; ++j)
      hessT[i][j] = hess_F[i][j];
  }

  inv_hessT.invert(hessT);
  inv_hessT.vmult(dkT, gradT);
  
  for (unsigned int i = 0; i < n_dofs; ++i)
    dk[i] = dkT(i);

  // somente alguns prints
  if(false)
  {
    // print da inversa
  std::cout << "\ninv_hessT:\n";
  for(unsigned int i = 0; i < n_dofs; ++i)
  {
    for(unsigned int j = 0; j < n_dofs; ++j)
      std::cout << inv_hessT[i][j] << "\t";
    std::cout << std::endl;
  }

  // print do gradiente
  std::cout << "\ngradT:\n";
  for(unsigned int i = 0; i < n_dofs; ++i)
    std::cout << gradT[i] << std::endl;

  // print do dk
  std::cout << "\ndk:\n";
  for(unsigned int i = 0; i < n_dofs; ++i)
    std::cout << dk[i] << std::endl;
  }
}

/* void Solver::compute_alfa()
{
  std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> new_s(n_dofs, 0); // new_s = s_k + alfa^(i) * d_k
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> alfa = 0;
  alfa.diff(0,1); // so vamos derivar com relacao a alfa
  alfa.val().diff(0,1); // mas vamos derivar 2 vezes

  // codigo duplicado (mas modificado) de compute_F_grad_hess:
  for (unsigned int i = 0; i < n_dofs; ++i) // diz que solution sao variaveis independentes
    new_s[i] = solution[i] + alfa * dk[i];

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
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        sg = 0;
        sg_prime = 0;
        double rho = fe_values.quadrature_point(q)[0]; // coordenada global
      
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double phi_i = fe_values.shape_value(i, q);
          const double phi_i_prime = fe_values.shape_grad(i, q)[0]; // acessa o seu unico elemento
          sg += new_s[local_dof_indices[i]] * phi_i;
          sg_prime += new_s[local_dof_indices[i]] * phi_i_prime;
        }

        E_h += (pow(rho*sg_prime, 2) + 2*gama*pow(sg,2)) * fe_values.JxW(q);
        P_h += pow(rho, 2) / ( (1+sg_prime) * pow(1+sg/rho, 2) - eps ) * fe_values.JxW(q);
      }
    } // end for cells

    E_h = c12 * pow(dofs.back(), 2) * radius + 
          pressure * dofs.back() * pow(radius, 2) +
          c11/2 * E_h;
    F_delta = E_h + P_h / delta;
      
    // montagem do gradiente e da hessiana
    for(unsigned int i = 0; i < n_dofs; ++i)
    {
      grad_F[i] = F_delta.dx(i).val();
      for(unsigned int j =0; j < n_dofs; ++j)
        hess_F[i][j] = F_delta.dx(i).dx(j);
    }

  
  // chama compute_F_hess_grad em uma versao modificada que nao calcula derivadas apenas o F_delta


  std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> dofs(n_dofs, 0);
  for (unsigned int i = 0; i < n_dofs; ++i) // diz que solution sao variaveis independentes
  {
    new_s[i] = solution[i];
    new_s[i].diff(i, n_dofs);
    new_s[i].val().diff(i, n_dofs);
  }
  
} */

void Solver::solve()
{
  compute_F_grad_hess();

  //compute_dk();

  //compute_alfa();

  // novo s = s + alfa * dk

  // print do gradiente
  std::cout << "\ngrad_F:\n";
  for(unsigned int i = 0; i < n_dofs; ++i)
    std::cout << grad_F[i] << std::endl;
  
  // print da hessiana
  std::cout << "\nhess_F:\n";
  for(unsigned int i = 0; i < n_dofs; ++i)
  {
    for(unsigned int j = 0; j < n_dofs; ++j)
      std::cout << hess_F[i][j] << "\t";
    std::cout << std::endl;
  }
}


void Solver::run ()
{
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
    n_dofs = dof_handler.n_dofs();
    //solution.resize(n_dofs, 0); // lembrar que primeiro dof é sempre 0
    solution = {-0.0001,-0.002,-0.005};
    grad_F.resize(n_dofs, 0);
    dk.resize(n_dofs, 0);
    hess_F.resize(n_dofs, grad_F);
    /* for (unsigned int i = 0; i < n_dofs; ++i) // diz que solution sao variaveis independentes
      {
        solution[i].diff(i, n_dofs);
        solution[i].val().diff(i, n_dofs);
      } */
      
    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
  
  solve();
  }
}