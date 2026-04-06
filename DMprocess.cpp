// GOAL: implement a parallel K-Mean clustering algorithm
// clusters are different genres.
// challenges: domain decomposition (splitting data among cores)
//             share centroids and update until convergence
//             generate output and visualization

// will be using MPI

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include "rapidcsv.h" // library for easily importing csvs


// helper function to generate a random float between two points.
float bounded_rand(int low, int high) {
    float width = high - low;
    return ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * width);
}


int main(void) {
    int comm_sz; // threadcount
    int my_rank; // rank
    srand(4);
    int CLUSTERS = 10;

    // initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    // process 0 reads file into memory
    if (my_rank == 0) {
        rapidcsv::Document source("super_culled_tracks_features.csv");
        // every included col (besides id) can be a float.
        std::vector<float> col = source.GetColumn<float>("loudness"); // ex
        
        // create some clusters (let's say ten)
        // every point in this space is a list of size 15.
        float clusters[CLUSTERS][15];
        for (int cluster = 0; cluster < CLUSTERS; cluster++) {
            clusters[cluster][0] = std::round(bounded_rand(0, 1));   // explicit 0/1
            clusters[cluster][1] = bounded_rand(0, 1);               // danceability 0-1
            clusters[cluster][2] = bounded_rand(0, 1);               // energy 0-1
            clusters[cluster][3] = std:round(bounded_rand(0, 11));   // key 0-11
            clusters[cluster][4] = bounded_rand(-60, 7.23);          // loudness -60-7.23 db (weird)
            clusters[cluster][5] = std::round(bounded_rand(0, 1));   // mode 0/1 minor/major
            
        }
        
    }

    // 1) k initial "means" are randomly generated within the data domain.
    // 2) k clusters are created by associating every observation with the nearest mean.
    // 3) The centroid of each of the k clusters becomes the new mean.
    // 4) Repeat steps 2 and 3 until convergence.



    MPI_Finalize();
    return 0;
}