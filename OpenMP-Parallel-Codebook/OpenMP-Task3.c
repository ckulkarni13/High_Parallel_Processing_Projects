/* CSYE7105 HW1 Q3: totals 14 points */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NRA 1000                 /* number of rows in matrix A */
#define NCA  500                 /* number of columns in matrix A */
#define NCB 1000                 /* number of columns in matrix B */

int main (int argc, char *argv[])
{
    int tid, nthreads, i, j, k, chunk;
    double a[NRA][NCA], /* matrix A to be multiplied */
           b[NCA][NCB], /* matrix B to be multiplied */
           c[NRA][NCB]; /* result matrix C */

    chunk = 10; /* set loop iteration chunk size */

    // Parallelize the following block using OpenMP directives with data scope: 10 points
    #pragma omp parallel private(tid, i, j, k) shared(a, b, c, chunk)
    {
        tid = omp_get_thread_num();

        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Starting matrix multiply example with %d threads\n", nthreads);
            printf("Initializing matrices...\n");
        }

        /*** Initialize matrices ***/
        
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= i+j;
        
        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                b[i][j]= i*j;

        for (i=0; i<NRA; i++)
            for (j=0; j<NCB; j++)
                c[i][j]= 0;

        printf("Thread %d starting matrix multiply...\n", tid);

        // scheduling the for-loop with 'static' kind using 'chunk': 4 points  
        #pragma omp for schedule(static, chunk)
        for (i=0; i<NRA; i++) 
        {

            for (j=0; j<NCB; j++)
                for (k=0; k<NCA; k++)
                    c[i][j] += a[i][k] * b[k][j];
        }
    } /*** End of parallel region ***/
    
}
