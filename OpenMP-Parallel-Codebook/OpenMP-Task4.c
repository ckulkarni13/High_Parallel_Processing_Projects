/* CSYE7105 HW1 Q4: totals 8 points */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define MASTER      0

int main (int argc, char *argv[])
{
    int numtasks, rank, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    // initialize MPI (2 points)
    MPI_Init(&argc, &argv);

    // get number of tasks (2 points)
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // get the rank (2 points)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // this one is obvious
    MPI_Get_processor_name(hostname, &len);

    printf("Hello from task %d on %s!\n", rank, hostname);

    if (rank == MASTER)
        printf("MASTER: Number of MPI tasks is: %d\n", numtasks);

    // close the parallel region (2 points)
    MPI_Finalize();

}
