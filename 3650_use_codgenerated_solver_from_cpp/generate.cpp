#include <stdio.h>
#include <dlfcn.h>


// C++ (and CasADi) from here on
#include <casadi/casadi.hpp>
using namespace casadi;

void usage_cplusplus(){
  std::cout << "---" << std::endl;
  std::cout << "Usage from CasADi C++:" << std::endl;
  std::cout << std::endl;

  // Use CasADi's "external" to load the compiled function
  Function f = external("f");

  // Use like any other CasADi function
  std::vector<double> x = {1, 2, 3, 4};
  std::vector<DM> arg = {reshape(DM(x), 2, 2), 5};
  std::vector<DM> res = f(arg);

  std::cout << "result (0): " << res.at(0) << std::endl;
  std::cout << "result (1): " << res.at(1) << std::endl;
}


int main(){
  // Variables
  SX x = SX::sym("x", 2, 2);
  SX y = SX::sym("y");

  // Simple function
  Function f("f", {x, y}, {sqrt(y)-1, sin(x)-y});

  // Generate C-code
  f.generate("f");

  // Compile the C-code to a shared library
  std::string compile_command = "gcc -fPIC -shared -O3 f.c -o f.so";
  int flag = system(compile_command.c_str());
  casadi_assert(flag==0, "Compilation failed");

  // Usage from C++
  usage_cplusplus();

  // Generate C-code
  f.generate("f_with_mem", {{"with_mem", true}});

  // Compile the C-code to a shared library
  compile_command = "gcc -fPIC -I/usr/include/ -shared -g f_with_mem.c -o f_with_mem.so";
  flag = system(compile_command.c_str());
  casadi_assert(flag==0, "Compilation failed");

  return 0;
}

