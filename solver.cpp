
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/lapack_full_matrix.h>

//#include <deal.II/lac/solver_cg.h>
//#include <deal.II/lac/precondition.h>
#include <iomanip> // @@@
#include <limits> //@@@

#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>

#include <Sacado.hpp>


#include "solver.h"


MySolver::MySolver(const int poly_degree, const unsigned int refine_global, const unsigned int quad_degree): 
  refine_global(refine_global),
  quad_degree(quad_degree),
  dof_handler(triangulation),
  fe(poly_degree)
{}

MySolver::~MySolver()
{}

void MySolver::compute_F_grad_hess()
{
  const bool analytic_diff = false;

  // reset gradiente e hessiana para zero
  grad_F.assign(grad_F.size(), 0);
  hess_F.assign(grad_F.size(), grad_F);
  

  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> F_delta, E_h, P_h, DE_h, DP_h, DDE_h;
  //std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> dofs(n_dofs, 0);

  // inicializacao de F_delta, E_h, ... , dofs
  F_delta = 0;
  E_h = 0;
  P_h = 0;
  DE_h = 0;
  DP_h = 0;
  /* for (unsigned int i = 0; i < n_dofs; ++i) // diz que dofs sao variaveis independentes
  {
    dofs[i] = solution[i];
    dofs[i].diff(i, n_dofs);
    dofs[i].val().diff(i, n_dofs);
  } */

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
    std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> dofs(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      dofs[i] = solution[local_dof_indices[i]];
      dofs[i].diff(i, dofs_per_cell);
      dofs[i].val().diff(i, dofs_per_cell); //antes estava n_dofs
    }

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      sg = 0;
      sg_prime = 0;
      double rho = fe_values.quadrature_point(q)[0]; // coordenada global
    
      // na verdade teria que passar por todos dofs, mas como phi é sempre zero fora da celula
      // so passamos pelos dofs da celula para calcular o produto escalar
      // lembrar que "g" é uma funcao de rho, entao ela vai fazer aprte da somatoria nos quad normalmente
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);
        const double phi_i_prime = fe_values.shape_grad(i, q)[0]; // acessa o seu unico elemento
        sg += dofs[i] * phi_i;
        sg_prime += dofs[i] * phi_i_prime;
      }

      E_h = 0; P_h = 0; F_delta = 0; // zera qualquer historico de contas
      E_h = (pow(rho*sg_prime, 2) + 2*gama*pow(sg,2)) * fe_values.JxW(q); //+=
      P_h = pow(rho, 2) / ( (1+sg_prime) * pow(1+sg/rho, 2) - eps ) * fe_values.JxW(q); //+=
      //P_h = pow(rho, 2) *  // alternativa com soma do det alem da soma do inverso do det
      //      (1 / ((1+sg_prime) * pow(1+sg/rho, 2) - eps) +
      //       ((1+sg_prime) * pow(1+sg/rho, 2) - eps) ) * fe_values.JxW(q);

      F_delta = c11/2 * E_h + P_h / delta;

      // adiciona contribuicao de uma das parcelas da somatoria
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        grad_F[local_dof_indices[i]] += F_delta.dx(i).val();
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          hess_F[local_dof_indices[i]][local_dof_indices[j]] += F_delta.dx(i).dx(j);
      }
        

      // Calculo derivadas analiticas só de eta_1 (para o eta_n falta adicionar termos ao DE_h)
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

  // adiciona a parcela da derivada que nao esta no somatorio de pontos de quadratura (só do eta_n)
  grad_F[n_dofs-1] += 2*c12*solution[n_dofs-1]*radius + pressure*pow(radius,2);
  hess_F[n_dofs-1][n_dofs-1] += 2 * c12 * radius;

  /* E_h = c12 * pow(dofs.back(), 2) * radius + 
        pressure * dofs.back() * pow(radius, 2) +
        c11/2 * E_h;
  F_delta = E_h + P_h / delta; */

  if(analytic_diff)
  {
    DE_h = c11/2 * DE_h;
    DDE_h = c11/2 * DDE_h;
  }
    
  // montagem do gradiente e da hessiana
  /* for(unsigned int i = 0; i < n_dofs; ++i)
  {
    grad_F[i] = F_delta.dx(i).val();
    for(unsigned int j =0; j < n_dofs; ++j)
      hess_F[i][j] = F_delta.dx(i).dx(j);
  } */

  // alguns prints para conferir valores
  if(false)
  {
    std::cout << "\nE_h: " << E_h << "\nP_h: " << P_h << "\nF_delta: " << F_delta 
              << "\nDE_h: " << DE_h <<  "\nDP_h: " << DP_h << "\nDDE_h: " << DDE_h << "\n";
  }
    
}

void MySolver::compute_dk()
{
  // cria o Tensor que vai conter o gradiente e a hessiana
  //Vector<double> gradT(grad_F.begin(), grad_F.end()), dkT(grad_F.begin(), grad_F.end());
  //dkT.reinit(n_dofs);
  //FullMatrix<double> hessT(n_dofs, n_dofs), inv_hessT(n_dofs, n_dofs);

  // Resolvendo o sistema com solver lapack
  // Na verdade o ideal seria trabalhar com a hessiana como uma SparseMatrix e usar
  // algum solver de matriz esparsa da dealii
  Vector<double> dkT(grad_F.begin(), grad_F.end());
  LAPACKFullMatrix<double> lapack_hess(n_dofs);
  
  for (unsigned int i = 0; i < n_dofs; ++i)
  {
    //gradT(i) = grad_F[i];
    for (unsigned int j = 0; j < n_dofs; ++j)
      //hessT[i][j] = hess_F[i][j];
      lapack_hess(i,j) = hess_F[i][j];
  }
  //==========================================
  //SolverControl solver_control(1000, 1e-12);
  //SolverCG<> solver(solver_control); //<> vazio indica que é o padrao: Vector<double>
  //PreconditionSSOR<> preconditioner; //<> vazio indica que é o padrao: SparseMatrix<double>
  //preconditioner.initialize(hessT, 1.2);
  //solver.solve(hessT, dkT, gradT, PreconditionIdentity());
  //==========================================
  
  // .solve espera que ja tenhamos feito a LU factorization
  // inicialmente dkT é o vetor do lado direito, mas apos o .solve ele vira a solucao do sistema
  lapack_hess.compute_lu_factorization();
  lapack_hess.solve(dkT);

  //inv_hessT.invert(hessT);
  //inv_hessT.vmult(dkT, gradT);
  
  // dk[0] nunca sera atualizado, entao dk[0] sera sempre zero, entao new_s[0] e solution[0] serao sempre zero
  for (unsigned int i = 1; i < n_dofs; ++i)
    dk[i] = -dkT(i); // d_k = - inv(hess)*grad

}

void MySolver::compute_alpha_derivs(double alpha, double &dF_dAlpha, double &d2F_dAlpha2)
{
  // alphaAD é uma variavel interna de compute_alpha_derivs que recebe o valor do alpha atual
  // e é usada para fazer as derivadas
  std::vector<Sacado::Fad::DFad<Sacado::Fad::DFad<double>>> new_s(n_dofs, 0); // new_s = s_k + alpha^(i) * d_k
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> alphaAD = alpha;
  alphaAD.diff(0,1); // so vamos derivar com relacao a alphaAD
  alphaAD.val().diff(0,1); // mas vamos derivar 2 vezes

  // codigo duplicado (mas modificado) de compute_F_grad_hess:
  Sacado::Fad::DFad<Sacado::Fad::DFad<double>> F_delta, E_h, P_h;
  F_delta = 0; E_h = 0; P_h = 0;
  for (unsigned int i = 0; i < n_dofs; ++i)
    new_s[i] = solution[i] + alphaAD * dk[i];

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

  E_h = c12 * pow(new_s.back(), 2) * radius + 
        pressure * new_s.back() * pow(radius, 2) +
        c11/2 * E_h;
  F_delta = E_h + P_h / delta;
    
  // derivada de F_delta com relacao a alpha
  dF_dAlpha = F_delta.dx(0).val();
  d2F_dAlpha2 = F_delta.dx(0).dx(0);
}

void MySolver::compute_alpha()
{
  double alpha = 0, prev_alpha = 0, // alpha(0) = 0
         dF_dAlpha, d2F_dAlpha2;
  std::vector<double> new_s(n_dofs, 0); // new_s = s_k + alpha^(i) * d_k

  for (unsigned int iter = 0; iter < iter_limit_alpha; ++iter)
  {
    // calcula as derivadas e poe em dF_dAlpha, d2F_dAlpha2
    compute_alpha_derivs(alpha, dF_dAlpha, d2F_dAlpha2);

    prev_alpha = alpha;
    alpha = alpha - dF_dAlpha / d2F_dAlpha2;

    if(verbose)
      std:: cout << "prev_alpha: " << prev_alpha << "   alpha: " << alpha << "\ndalpha: " << dF_dAlpha 
                 << "\nd2F_dAlpha2: " << d2F_dAlpha2 << "\n";

    // atualiza new_s com o novo alpha e o mesmo s_k (solution) e d_k
    for (unsigned int i = 0; i < n_dofs; ++i)
      new_s[i] = solution[i] + alpha * dk[i];

    // confere se viola (70) em algum dos nos
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    typename DoFHandler<1>::active_cell_iterator
      cell = dof_handler.begin_active(),  endc = dof_handler.end();
    for (; cell!=endc; ) // rodo ++cell manualmente
    {
      cell->get_dof_indices(local_dof_indices);
      //double rho = cell->vertex(1)(0);
      double rho_m = (cell->vertex(1)(0) + cell->vertex(0)(0))*0.5; // rho no meio da celula
      double h = cell->vertex(1)(0) - cell->vertex(0)(0);
      double phi_i_prime = 1 / h;
      double sg_prime = new_s[local_dof_indices[0]]*(-phi_i_prime) + new_s[local_dof_indices[1]]*phi_i_prime;
      //double sg = new_s[local_dof_indices[1]]; //sg de rho
      double sg = new_s[local_dof_indices[0]]*0.5 + new_s[local_dof_indices[1]]*0.5; //sg de rho_m
      //double det = (1 + sg_prime) * pow(1 + sg/rho, 2) - eps;
      double det = (1 + sg_prime) * pow(1 + sg/rho_m, 2) - eps;


      if(det < 0)
      {
        if(verbose)
          std::cout << "  Aviso: alpha deu um pulo muito grande, reduzindo seu valor e reavaliando (70) nos nós...\n";
        alpha = alpha / 2;
        for (unsigned int i = 0; i < n_dofs; ++i)
          new_s[i] = solution[i] + alpha * dk[i];
        cell = dof_handler.begin_active();
      }
      else
        ++cell;
      
    }

    if(std::abs((alpha - prev_alpha)/alpha) < alpha_tol)
    {
      if(verbose)
        std::cout << "\nsaindo...  alpha update: " << std::abs((alpha - prev_alpha)/alpha) << "\n";
      break; // sai do loop do alpha
    }
      
    if(iter == iter_limit_alpha - 1)
      std::cout << "\n   Aviso: loop do alpha atingiu o limite de iteracoes e foi aceito como alpha final.\n";
  }
  alpha_k = alpha;
  /* // ja tenho que ter declarado fe_values, quadrature_formula,...
    // usar QMidpoint para calcular sempre no meio da celula
    // passar esse QMidpoint para o fe_values que calcula phi_linha
    // e passar QTrapz para o fe_values que calcula phi
    fe_values_trapez.reinit(cell);
    fe_values_midpoint.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    sg = 0;
    sg_prime = 0;
    double rho = cell->vertex(1)(0); // espero que o segundo vertice seja sempre o do rho maior [CONFERIR]
    double h = cell->vertex(1)(0) - cell->vertex(0)(0);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      // QMidpoint so tem 1 ponto de quadratura e QTrapez so tem 2 (caso dim=1)
      // talvez calculando com o JxW e sabendo o valor do W, eu consigo fazer a transferencia dos valores
      // calculados pelos fe_values na celula de referencia e o valor que seria calculado na celula real
      // outra opcao é usar (dof_2-dof_1)/h para a derivada e 1 ou 0 para o phi, ja que os phis sao lineares
      //const double phi_i = fe_values_trapez.shape_value(i, 1) * fe_values_trapez.JxW(1) * 2; // W = 0.5
      //const double phi_i_prime = fe_values_midpoint.shape_grad(i, 0)[0] * fe_values_midpoint.JxW(0); // W = 1 
      sg += new_s[local_dof_indices[i]] * phi_i;
      sg_prime += new_s[local_dof_indices[i]] * phi_i_prime;
    } */


  
}

void MySolver::solve()
{
  // so para "declarar" t1 e t2
  auto t1 = std::chrono::high_resolution_clock::now();
  auto t2 = std::chrono::high_resolution_clock::now();

  for (unsigned int iter_delta = 0; iter_delta < 50; ++iter_delta)
  {
    for (unsigned int iter_sk = 0; iter_sk < iter_limit_sk; ++iter_sk)
    {
      t1 = std::chrono::high_resolution_clock::now();
      compute_F_grad_hess(); // usa solution e atualiza grad_F, hess_F
      t2 = std::chrono::high_resolution_clock::now();
      timing[0] += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

      t1 = std::chrono::high_resolution_clock::now();
      compute_dk(); // usa grad_F e hess_F e atualiza dk
      t2 = std::chrono::high_resolution_clock::now();
      timing[1] += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

      t1 = std::chrono::high_resolution_clock::now();
      compute_alpha(); // usa solution e dk e atualiza alpha_k
      t2 = std::chrono::high_resolution_clock::now();
      timing[2] += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

      t1 = std::chrono::high_resolution_clock::now();
      // atualiza solution
      prev_solution = solution;
      for (unsigned int i = 0; i < n_dofs; ++i)
        solution[i] = solution[i] + alpha_k * dk[i];

      // criterio de parada para a serie de s_k
      double solution_crit, solution_sum = 0, prev_solution_sum = 0;
      for (unsigned int i = 0; i < n_dofs; ++i)
      {
        solution_sum += std::abs(solution[i]);
        prev_solution_sum += std::abs(prev_solution[i]);
      }
      solution_crit = (solution_sum - prev_solution_sum) / (solution_sum + 1e-10);
      if (solution_crit < solution_tol)
      {
        if(verbose)
          std::cout << "\nsaindo...  solution update: " << solution_crit << "\n";
        break; // sai do loop do s_k
      }

      if(iter_sk == iter_limit_sk - 1)
        std::cout << "\n   Aviso: loop do s_k atingiu o limite de iteracoes e foi aceito como s_k final.\n";

      t2 = std::chrono::high_resolution_clock::now();
      timing[3] += std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    }
    
    // insere linha = {delta, dof0, dof1, dof2, ..., dofN}
    output_data.emplace_back(solution);
    output_data[iter_delta].insert(output_data[iter_delta].begin(), delta);

    //output_file << "\nSolution para delta = " << delta << "\n";
    //for(unsigned int i = 0; i < n_dofs; ++i)
        //output_file << solution[i] << std::endl;
    
    delta = delta * 10;
    if (delta > delta_max)
      break;
  }
  

  // so alguns prints
  if(false)
  {
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

    // print da solucao
    std::cout << "\nsolution:\n";
    for(unsigned int i = 0; i < n_dofs; ++i)
      std::cout << solution[i] << std::endl;
  }
  
}

void MySolver::create_480_cells()
{
  std::vector<std::vector<double>> cell_sizes(1);
  std::vector<double> tmp(480, 0.);
  Point<1> origin(0), end(1);
  cell_sizes[0] = tmp;

  for(unsigned int i=0; i<480; ++i)
  {
    if (i<300) cell_sizes[0][i] = 0.07/300;
    else if (i>=300 && i<400) cell_sizes[0][i] = 0.39/100;
    else cell_sizes[0][i] = 0.54/80;
  }
  GridGenerator::subdivided_hyper_rectangle(triangulation, cell_sizes, origin, end, true);

  /* std::cout << "RHOS\n";
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      double rho = cell->vertex(1)(0);
      std::cout << rho << "\n";
    } */
} // end create_480_cells()

void MySolver::write_output_file(const unsigned int cycle)
{
  output_file.open("out/sol ref" + std::to_string(refine_global) +
                   "-" + std::to_string(cycle) + ".txt");
  
  // add os rho de cada dof na primeira linha de output_data
  std::vector<double> rhos(n_dofs, 0.);
  int i_rhos = 1; // primeiro rho é zero
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      rhos[i_rhos] = cell->vertex(1)(0);
      i_rhos += 1;
    }
    rhos.insert(rhos.begin(), 0); // para alinhar com os deslocs
    output_data.insert(output_data.begin(), rhos);

  for (unsigned int i = 0; i < size(output_data); ++i)
  {
    for (unsigned int j = 0; j < size(output_data[i]); ++j)
    {
      if(j == size(output_data[i])-1) {
        output_file << output_data[i][j] << std::endl;
      } else {
        output_file << output_data[i][j] << ";";
      }
    }
  }
  output_file.close();
  std::cout << "Arquivo " << "'out/sol ref" << refine_global << ".txt' gerado!" << std::endl;
  
} // end write_output_file()

void MySolver::compute_lagrange_det()
{
  // calculo determinante ao longo do raio
  std::vector<double> dets;
  std::vector<types::global_dof_index> local_dof_indices (2);
  for(const auto &cell : dof_handler.active_cell_iterators())
  {
    cell->get_dof_indices(local_dof_indices);
    //double rho = cell->vertex(1)(0); // rho no meio da celula
    double rho_m = (cell->vertex(1)(0) + cell->vertex(0)(0))*0.5; // rho no meio da celula
    double h = cell->vertex(1)(0) - cell->vertex(0)(0);
    double phi_i_prime = 1 / h;
    double sg_prime = solution[local_dof_indices[0]]*(-phi_i_prime) + solution[local_dof_indices[1]]*phi_i_prime;
    //double sg = solution[local_dof_indices[1]]; //sg de rho
    double sg = solution[local_dof_indices[0]]*0.5 + solution[local_dof_indices[1]]*0.5; //sg de rho_m
    //double det = (1 + sg_prime) * pow(1 + sg/rho, 2);
    double det = (1 + sg_prime) * pow(1 + sg/rho_m, 2);
    dets.emplace_back(det);

    //std::cout << std::setprecision (std::numeric_limits<double>::max_digits10)
    //          << det
    //          << std::endl;
  }
  //std::cout << std::setprecision (std::numeric_limits<double>::max_digits10)
  //            << eps
  //            << std::endl;
  
  std::vector<double> lagranges;
  for(unsigned int i=0; i<size(dets); ++i)
  {
    // para delta = 1e+8
    double lagrange = 1. / (1e+8 * pow(dets[i]-eps, 2));
    lagranges.emplace_back(lagrange);
  }

  dets.insert(dets.begin(), 0); // para alinhar com os rhos
  dets.insert(dets.begin(), 0);
  output_data.emplace_back(dets);

  lagranges.insert(lagranges.begin(), 0); // para alinhar com os rhos
  lagranges.insert(lagranges.begin(), 0);
  output_data.emplace_back(lagranges);

}

void MySolver::run ()
{
  for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;
    pressure = final_pressure*(cycle+1)/n_cycles;
    delta = delta_min;
    output_data.clear();
    if (cycle == 0)
    {
      if(refine_global == 0) create_480_cells();
      else {
        GridGenerator::hyper_cube(triangulation, 0, radius, /*colorize*/ true);
        triangulation.refine_global(refine_global);
      }

      std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells()
              << std::endl;

      dof_handler.distribute_dofs(fe); // garantir numeracao do rho=0 ate o rho=radius
      n_dofs = dof_handler.n_dofs();
      solution.resize(n_dofs, 0); // lembrar que primeiro dof é sempre 0, faço isso deixando sempre dk[0]=0
      //solution = {-0.0001,-0.002,-0.005};
      prev_solution.resize(n_dofs, 0);
      grad_F.resize(n_dofs, 0);
      dk.resize(n_dofs, 0);
      hess_F.resize(n_dofs, grad_F);

      std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
    }
    //else
    //  Assert(false, ExcNotImplemented()); //refine_grid ();

    
    alpha_k = 0;
      
    
    
    // [UPDATE] nao precisa mais desse Renumbering porque refiz o create_480_cells
    // Esse Renumbering faz os dofs menores (0,1,3,4,...) estarem proximos de rho=0
    // e os dofs maiores estarem proximos de rho=radius
    // isso é necessario para a malha nao uniforme, para a uniforme nao muda nada
    // um motivo que eu sei que me obriga a deixar os dofs na ordem ao longo de rho é que
    // o meu codigo esta considerando que o dof=0 é o dof fixo, entao necessariamente tenho
    // que ter o dof numero 0 em rho=0
    // pode tambem ter outros motivos, nao conferi
    // mais info em:
    //  https://www.dealii.org/current/doxygen/deal.II/namespaceDoFRenumbering.html#a6ad7b76064dd49a98187ff2b06298cf9
    //  https://www.dealii.org/current/doxygen/deal.II/DEALGlossary.html#GlossZOrder
    //DoFRenumbering::hierarchical(dof_handler);

    // esse loop comentado é so para verificar que os dofs estao ordenados
    /* std::vector<types::global_dof_index> local_dof_indices (2);
    std::cout << "Coordenadas e graus de liberdade de cada celula" << "\n";
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices (local_dof_indices);
      double rho1 = cell->vertex(1)(0);
      double rho0 = cell->vertex(0)(0);
      std::cout << rho1 << " " << rho0 << " " << rho1-rho0 << " "
                << local_dof_indices[0] << " " << local_dof_indices[1] << std::endl;
    } */

    //output_file.open("out/sol ref" + std::to_string(refine_global) + ".txt");
    solve();
    //output_file.close();
    compute_lagrange_det();
    write_output_file(cycle);

    std::cout << "timing:" << std::endl << timing[0]  << std::endl << timing[1] 
              << std::endl << timing[2] << std::endl << timing[3] << std::endl;
    
  }
}