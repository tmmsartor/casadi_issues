#include <stdio.h>
#include <dlfcn.h>
#include <casadi/mem.h>

int main(){
  printf("---\n");
  printf("Standalone usage from C/C++:\n");
  printf("\n");

  /* Handle to the dll */
  void* handle;

  /* Load the dll */
  handle = dlopen("./f.so", RTLD_LAZY);
  if(handle==0){
    printf("Cannot open f.so, error %s\n", dlerror());
    return 1;
  }

  /* Reset error */
  dlerror();

  /*typedef long long int casadi_int;*/

  /* Typedefs */
  typedef void (*signal_t)(void);
  typedef casadi_int (*getint_t)(void);
  typedef int (*work_t)(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
  typedef const casadi_int* (*sparsity_t)(casadi_int ind);
  typedef int (*eval_t)(const double** arg, double** res, casadi_int* iw, double* w, int mem);
  typedef int (*casadi_checkout_t)(void);
  typedef void (*casadi_release_t)(int);

  /* Memory management -- increase reference counter */
  signal_t incref = (signal_t)dlsym(handle, "f_incref");
  if(dlerror()) dlerror(); // No such function, reset error flags

  /* Memory management -- decrease reference counter */
  signal_t decref = (signal_t)dlsym(handle, "f_decref");
  if(dlerror()) dlerror(); // No such function, reset error flags

  /* Thread-local memory management -- checkout memory */
  casadi_checkout_t checkout = (casadi_checkout_t)dlsym(handle, "f_checkout");
  if(dlerror()) dlerror(); // No such function, reset error flags

  /* Thread-local memory management -- release memory */
  casadi_release_t release = (casadi_release_t)dlsym(handle, "f_release");
  if(dlerror()) dlerror(); // No such function, reset error flags

  /* Number of inputs */
  getint_t n_in_fcn = (getint_t)dlsym(handle, "f_n_in");
  if (dlerror()) return 1;
  casadi_int n_in = n_in_fcn();

  /* Number of outputs */
  getint_t n_out_fcn = (getint_t)dlsym(handle, "f_n_out");
  if (dlerror()) return 1;
  casadi_int n_out = n_out_fcn();

  /* Get sizes of the required work vectors */
  casadi_int sz_arg=n_in, sz_res=n_out, sz_iw=0, sz_w=0;
  work_t work = (work_t)dlsym(handle, "f_work");
  if(dlerror()) dlerror(); // No such function, reset error flags
  if (work && work(&sz_arg, &sz_res, &sz_iw, &sz_w)) return 1;
  printf("Work vector sizes:\n");
  printf("sz_arg = %lld, sz_res = %lld, sz_iw = %lld, sz_w = %lld\n\n",
         sz_arg, sz_res, sz_iw, sz_w);

  /* Input sparsities */
  sparsity_t sparsity_in = (sparsity_t)dlsym(handle, "f_sparsity_in");
  if (dlerror()) return 1;

  /* Output sparsities */
  sparsity_t sparsity_out = (sparsity_t)dlsym(handle, "f_sparsity_out");
  if (dlerror()) return 1;

  /* Print the sparsities of the inputs and outputs */
  casadi_int i;
  for(i=0; i<n_in + n_out; ++i){
    // Retrieve the sparsity pattern - CasADi uses column compressed storage (CCS)
    const casadi_int *sp_i;
    if (i<n_in) {
      printf("Input %lld\n", i);
      sp_i = sparsity_in(i);
    } else {
      printf("Output %lld\n", i-n_in);
      sp_i = sparsity_out(i-n_in);
    }
    if (sp_i==0) return 1;
    casadi_int nrow = *sp_i++; /* Number of rows */
    casadi_int ncol = *sp_i++; /* Number of columns */
    const casadi_int *colind = sp_i; /* Column offsets */
    const casadi_int *row = sp_i + ncol+1; /* Row nonzero */
    casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

    /* Print the pattern */
    printf("  Dimension: %lld-by-%lld (%lld nonzeros)\n", nrow, ncol, nnz);
    printf("  Nonzeros: {");
    casadi_int rr,cc,el;
    for(cc=0; cc<ncol; ++cc){                    /* loop over columns */
      for(el=colind[cc]; el<colind[cc+1]; ++el){ /* loop over the nonzeros entries of the column */
        if(el!=0) printf(", ");                  /* Separate the entries */
        rr = row[el];                            /* Get the row */
        printf("{%lld,%lld}",rr,cc);                 /* Print the nonzero */
      }
    }
    printf("}\n\n");
  }

  /* Function for numerical evaluation */
  eval_t eval = (eval_t)dlsym(handle, "f");
  if(dlerror()){
    printf("Failed to retrieve \"f\" function.\n");
    return 1;
  }

  /* Allocate input/output buffers and work vectors*/
  const double *arg[sz_arg];
  double *res[sz_res];
  casadi_int iw[sz_iw];
  double w[sz_w];

  /* Function input and output */
  const double x_val[] = {1,2,3,4};
  const double y_val = 5;
  double res0;
  double res1[4];

  // Allocate memory (thread-safe)
  incref();

  /* Evaluate the function */
  arg[0] = x_val;
  arg[1] = &y_val;
  res[0] = &res0;
  res[1] = res1;

  // Checkout thread-local memory (not thread-safe)
  // Note MAX_NUM_THREADS
  int mem = checkout();

  // Evaluation is thread-safe
  if (eval(arg, res, iw, w, mem)) return 1;

  // Release thread-local (not thread-safe)
  release(mem);

  /* Print result of evaluation */
  printf("result (0): %g\n",res0);
  printf("result (1): [%g,%g;%g,%g]\n",res1[0],res1[1],res1[2],res1[3]);

  /* Free memory (thread-safe) */
  decref();

  /* Free the handle */
  dlclose(handle);

  return 0;
}
