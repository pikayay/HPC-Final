// GOAL: implement a parallel K-Mean clustering algorithm
// clusters are different genres.
// challenges: domain decomposition (splitting data among cores)
//             share centroids and update until convergence
//             generate output and visualization

// will be using MPI
// note compile with mpic++ file -o out, run with mpiexec -n <threads> -ppn <processes-per-node> <file>

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
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
    const int CLUSTERS = 10;
    const int features_count = 15;
    const int MAX_ITERATIONS = 20; // limit needed until convergence is figurd out

    // hardcoded max distances for features, for normalization. may need tweaking.
    // loudness in particular; isn't dB logarithmic or something?
    // duration_ms i've truncated at 1mil instead of 6mil to weight it slightly higher
    // the vast majority of the songs are between 1 and 500000ms. difficult to normalize.
    // considering logging it.
    const float featureRanges[features_count] = {1, 1, 1, 11, 67, 1, 1, 1, 1, 1, 1, 249, 1000000, 5, 124};

    // vars for distributing data
    int total_rows = 0;
    std::vector<float> all_data; // big vector, populated on p0
    float clusters[CLUSTERS][features_count]; // shared centroids

    // initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    // process 0 reads file into memory
    if (my_rank == 0) {
        rapidcsv::Document source("super_culled_tracks_features.csv");
        total_rows = source.GetRowCount();

        // features extracted, in order
        std::vector<std::string> feature_names = {
            "explicit", "danceability", "energy", "key", "loudness", "mode", "speechiness", 
            "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", 
            "time_signature", "year"
        };
        
        printf("Read %d rows\n", total_rows);
        
        // squish that nice data format down into a 1d vector
        // looks like [ex, da, en, ke, lo, mo, sp, ac, in, li, va, tp, du, ts, yr...]
        // so a song would be 15 elements in a row, then the next 15 are another song.
        // doing this for better MPI_Send functionality
        all_data.resize(total_rows * features_count);
        for (int f = 0; f < features_count; f++) {
            std::vector<float> col = source.GetColumn<float>(feature_names[f]);
            for (int r = 0; r < total_rows; r++) {
                all_data[r * features_count + f] = col[r];
            }
        }
        
        // create some clusters
        // every point in this space is a list of size 15.
        for (int cluster = 0; cluster < CLUSTERS; cluster++) {
            clusters[cluster][0] = std::round(bounded_rand(0, 1));          // explicit 0/1
            clusters[cluster][1] = bounded_rand(0, 1);                      // danceability 0-1
            clusters[cluster][2] = bounded_rand(0, 1);                      // energy 0-1
            clusters[cluster][3] = std::round(bounded_rand(0, 11));         // key 0-11
            clusters[cluster][4] = bounded_rand(-60, 7.23);                 // loudness -60-7.23 db (weird)
            clusters[cluster][5] = std::round(bounded_rand(0, 1));          // mode 0/1 minor/major
            clusters[cluster][6] = bounded_rand(0, 1);                      // "speechiness" 0-1
            clusters[cluster][7] = bounded_rand(0, 1);                      // "acousticness" 0-1
            clusters[cluster][8] = bounded_rand(0, 1);                      // "instrumentalness" 0-1
            clusters[cluster][9] = bounded_rand(0, 1);                      // "liveness" 0-1
            clusters[cluster][10] = bounded_rand(0, 1);                     // valence 0-1
            clusters[cluster][11] = bounded_rand(0, 249);                   // tempo 0-249
            clusters[cluster][12] = bounded_rand(1, 6000000);               // duration 0-6m ms
            clusters[cluster][13] = std::round(bounded_rand(0, 5));         // time signature (vast majority 3/4)
            clusters[cluster][14] = std::round(bounded_rand(1900, 2024));   // year
        }
    }

    // broadcast the total number of rows and initial clusters to all ranks
    MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(clusters, CLUSTERS * features_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // calculate row distribution for this process (divide fairly)
    int base_rows = total_rows / comm_sz;
    int remainder = total_rows % comm_sz;
    int my_rows = base_rows + (my_rank < remainder ? 1 : 0);
    
    // allocate memory for this process's local chunk of data
    std::vector<float> my_data(my_rows * features_count);

    // for every row, store its cluster assignments
    std::vector<int> song_assignments(my_rows);

    if (my_rank == 0) {
        int offset = my_rows * features_count; // start sending after rank 0's portion

        // now process 0 needs to send chunks of memory to different processes
        for (int i = 1; i < comm_sz; i++) {
            int rows_for_i = base_rows + (i < remainder ? 1 : 0);
            int size_for_i = rows_for_i * features_count;
            MPI_Send(all_data.data() + offset, size_for_i, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            offset += size_for_i;
        }
        // rank 0 copies its own portion locally
        std::copy(all_data.begin(), all_data.begin() + (my_rows * features_count), my_data.begin());
    } else {
        // worker processes receive their chunks from rank 0
        MPI_Recv(my_data.data(), my_rows * features_count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // for every song (15 entries in my_data), compute the nearest cluster.
    }

    // 1) k initial "means" are randomly generated within the data domain. (Done)
    // 2) k clusters are created by associating every observation with the nearest mean.
    // 3) The centroid of each of the k clusters becomes the new mean.
    // 4) Repeat steps 2 and 3 until convergence.

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        // These are for calculating the new centroids. We need local sums and counts.
        std::vector<float> local_cluster_sums(CLUSTERS * features_count, 0.0f);
        std::vector<int> local_cluster_counts(CLUSTERS, 0);

        // --- STEP 2: ASSIGNMENT ---
        // Each process assigns its local points to the nearest cluster.
        for (int rowID = 0; rowID < my_rows; rowID++) {
            float cluster_distances[CLUSTERS];
            float min_dist = 9999;
            // for every cluster, compute the distances.
            float min_dist_sq = std::numeric_limits<float>::max();
            int assigned_cluster = -1;

            // Find the nearest cluster for the current data point.
            for (int clusterID = 0; clusterID < CLUSTERS; clusterID++) {
                float dist = 0;
                // for every feature in the cluster, compute distances, normalize, and add to total dist
                float dist_sq = 0;
                for (int featureID = 0; featureID < features_count; featureID++) {
                    // this will be between 0 and 1 ideally. 
                    float diff = (my_data[rowID * features_count + featureID] - clusters[clusterID][featureID]) / featureRanges[featureID];
                    dist += diff * diff; // distances are squared to avoid pos/neg annoyances
                    dist_sq += diff * diff;
                }

                // if this is the closest cluster, classify the song as such
                if (dist < min_dist) {
                    min_dist = dist;
                    song_assignments[rowID] = clusterID;
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    assigned_cluster = clusterID;
                }
            }
            song_assignments[rowID] = assigned_cluster;

            // add this song's values to the cluster's running sums to average later
            // Add this point's data to the local sums for its assigned cluster.
            local_cluster_counts[assigned_cluster]++;
            for (int featureID = 0; featureID < features_count; featureID++) {
                cluster_sums[song_assignments[rowID]][featureID] += my_data[rowID * features_count + featureID];
                local_cluster_sums[assigned_cluster * features_count + featureID] += my_data[rowID * features_count + featureID];
            }
        }

        // average running sums for the new centroids
        for (int clusterID = 0; clusterID < CLUSTERS; clusterID++) {
            for (int featureID = 0; featureID < features_count; featureID++) {
                cluster_sums[clusterID][featureID] /= my_rows;
        // --- STEP 3: UPDATE ---
        // Aggregate the local sums and counts to get global sums and counts.
        std::vector<float> global_cluster_sums(CLUSTERS * features_count, 0.0f);
        std::vector<int> global_cluster_counts(CLUSTERS, 0);
        MPI_Allreduce(local_cluster_sums.data(), global_cluster_sums.data(), CLUSTERS * features_count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cluster_counts.data(), global_cluster_counts.data(), CLUSTERS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Now, update the centroids based on the global aggregates.
        for (int c = 0; c < CLUSTERS; ++c) {
            if (global_cluster_counts[c] > 0) { // Avoid division by zero for empty clusters
                for (int f = 0; f < features_count; ++f) {
                    clusters[c][f] = global_cluster_sums[c * features_count + f] / global_cluster_counts[c];
                }
            }
            // Optional: Handle empty clusters, e.g., by re-initializing them.
        }

        // send new centroids back to thread 0.
        MPI_Send(cluster_sums, CLUSTERS * features_count, MPI_FLOAT, 1, my_rank, MPI_COMM_WORLD);
    }

    // 1) k initial "means" are randomly generated within the data domain.
    // 2) k clusters are created by associating every observation with the nearest mean.
    // 3) The centroid of each of the k clusters becomes the new mean.
    // 4) Repeat steps 2 and 3 until convergence.



    MPI_Finalize();
    return 0;
}