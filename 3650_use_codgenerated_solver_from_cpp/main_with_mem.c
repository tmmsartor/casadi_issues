#include <stdio.h>
#include <dlfcn.h>
#include <casadi/mem.h>

int main(){
  printf("---\n");
  printf("Usage from C/C++ with casadi/mem.h:\n");
  printf("\n");

  /* Handle to the dll */
  void* handle;

  /* Load the dll */
  handle = dlopen("./f_with_mem.so", RTLD_LAZY);
  if(handle==0){
    printf("Cannot open f_with_mem.so, error %s\n", dlerror());
    return 1;
  }

  /* Reset error */
  dlerror();

  /* Typedefs */
  typedef casadi_functions* (*functions_t)(void);

  /* mem.h interface */
  functions_t functions = (functions_t)dlsym(handle, "f_functions");
  if(dlerror()) dlerror(); // No such function, reset error flags

  casadi_functions* f = functions();

  casadi_mem* mem = casadi_alloc(f);

  /* Function input and output */
  const double x_val[] = {1,2,3,4};
  const double y_val = 5;
  double res0;
  double res1[4];

  /* Evaluate the function */
  mem->arg[0] = x_val;
  mem->arg[1] = &y_val;
  mem->res[0] = &res0;
  mem->res[1] = res1;

  casadi_eval(mem);

  /* Print result of evaluation */
  printf("result (0): %g\n",res0);
  printf("result (1): [%g,%g;%g,%g]\n",res1[0],res1[1],res1[2],res1[3]);

  casadi_free(mem);

  /* Success */
  return 0;
}

