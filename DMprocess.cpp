// GOAL: implement a parallel K-Mean clustering algorithm
// clusters are different genres.
// challenges: domain decomposition (splitting data among cores)
//             share centroids and update until convergence
//             generate output and visualization

// will be using MPI

#include <stdio.h>
#include <string.h>
#include <mpi.h>


int main(void) {
    int comm_sz; // threadcount
    int my_rank; // rank

    // initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    // process 0 reads file into memory
    if (my_rank == 0) {
        
    }

    MPI_Finalize();
    return 0;
}