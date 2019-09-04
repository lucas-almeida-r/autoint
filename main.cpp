
#include <fstream>
#include <iostream>
#include <cmath>
#include <Sacado.hpp>

#include "solver.h"

int main()
{
  try
    {
      if(false)
      {
        Sacado::Fad::DFad<double> a,b,c,d,e,f;
      //double d;
      a = 1; b = 2; c=0;
      a.diff(0,2);  // Set a to be dof 0, in a 2-dof system.
      b.diff(1,2);  // Set b to be dof 1, in a 2-dof system.
      d = 2*a;
      c = d*a;
      e=5*d;
      d=0;
      d=3*b;
      c += d*b;
      e+=7*d;
      d=0;
      f=c+e/2;
      //c += pow(a*b,2);
      //double *derivs = &f.fastAccessDx(0); // Access derivatives
      std::cout << "dc/da = " << f.dx(0) << ", dc/db=" << f.dx(1) << std::endl;

      c = 0,
      c += 2*a;
      c = cos(a*b) +c;
      //double *derivs = &c.fastAccessDx(0); // Access derivatives
      //std::cout << "dc/da = " << derivs[0] << ", dc/db=" << derivs[1] << std::endl;
      }
      

      Solver solver(1,4,3);
      solver.run();      

    } //end try
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
