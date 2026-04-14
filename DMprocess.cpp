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
#include <fstream>
#include "rapidcsv.h" // library for easily importing csvs


// helper function to generate a random float between two points.
float bounded_rand(float low, float high) {
    float range = high - low;
    return low + ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * range);
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

        all_data.resize(total_rows * features_count);

        // preload everything from the csv into columns of features
        std::vector<std::vector<float>> data_by_column;
        data_by_column.reserve(features_count);
        for (const auto& name : feature_names) {
            // special data handling for string field; could also clean data beforehand
            if (name == "explicit") {
                std::vector<std::string> explicit_str_col = source.GetColumn<std::string>(name);
                std::vector<float> explicit_float_col;
                explicit_float_col.reserve(explicit_str_col.size());
                // convert string booleans to 1.0f or 0.0f
                for (const std::string& val : explicit_str_col) {
                    explicit_float_col.push_back((val == "True") ? 1.0f : 0.0f);
                }
                data_by_column.push_back(explicit_float_col);
            } else {
                // everything else is numeric by default
                data_by_column.push_back(source.GetColumn<float>(name));
            }
        }

        // squish that nice data format down into a 1d vector
        // [exp, dance, energy, key, loud, mode, speech, acoustic, instr, live, valence, tempo, duration, ts, yr...]
        // so a song would be 15 elements in a row, then the next 15 are another song.
        // doing this for better MPI_Send functionality.
        for (int r = 0; r < total_rows; r++) {
            for (int f = 0; f < features_count; f++) {
                all_data[r * features_count + f] = data_by_column[f][r];
            }
        }
        
        printf("Read %d rows\n", total_rows);
        
        // create some clusters
        // every point in this space is a list of size 15.
        for (int cluster = 0; cluster < CLUSTERS; cluster++) {
            clusters[cluster][0] = std::round(bounded_rand(0.0f, 1.0f));         // explicit 0/1
            clusters[cluster][1] = bounded_rand(0.0f, 1.0f);                     // danceability 0-1
            clusters[cluster][2] = bounded_rand(0.0f, 1.0f);                     // energy 0-1
            clusters[cluster][3] = std::round(bounded_rand(0.0f, 11.0f));        // key 0-11
            clusters[cluster][4] = bounded_rand(-60.0f, 7.23f);                  // loudness -60-7.23 db (weird)
            clusters[cluster][5] = std::round(bounded_rand(0.0f, 1.0f));         // mode 0/1 minor/major
            clusters[cluster][6] = bounded_rand(0.0f, 1.0f);                     // "speechiness" 0-1
            clusters[cluster][7] = bounded_rand(0.0f, 1.0f);                     // "acousticness" 0-1
            clusters[cluster][8] = bounded_rand(0.0f, 1.0f);                     // "instrumentalness" 0-1
            clusters[cluster][9] = bounded_rand(0.0f, 1.0f);                     // "liveness" 0-1
            clusters[cluster][10] = bounded_rand(0.0f, 1.0f);                    // valence 0-1
            clusters[cluster][11] = bounded_rand(0.0f, 249.0f);                  // tempo 0-249
            clusters[cluster][12] = bounded_rand(1.0f, 6000000.0f);              // duration 0-6m ms
            clusters[cluster][13] = std::round(bounded_rand(0.0f, 5.0f));        // time signature (vast majority 3/4)
            clusters[cluster][14] = std::round(bounded_rand(1900.0f, 2024.0f));  // year
        }
    }

    // broadcast the total number of rows and initial clusters to all ranks
    MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(clusters, CLUSTERS * features_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) printf("Initial clusters broadcasted.\n");

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
        printf("Data distributed to all processes.\n");
    } else {
        // worker processes receive their chunks from rank 0
        MPI_Recv(my_data.data(), my_rows * features_count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    // steps 2-4, now that the data's been distributed:
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // local sums and counts to calculate new centroids later
        std::vector<float> local_cluster_sums(CLUSTERS * features_count, 0.0f);
        std::vector<int> local_cluster_counts(CLUSTERS, 0);

        // STEP 2 ----
        // each process assigns its local points to the nearest cluster.
        for (int rowID = 0; rowID < my_rows; rowID++) {
            float min_dist_sq = std::numeric_limits<float>::max();
            int assigned_cluster = -1;

            // find the nearest cluster for the current data point.
            for (int clusterID = 0; clusterID < CLUSTERS; clusterID++) {
                float dist_sq = 0;
                for (int featureID = 0; featureID < features_count; featureID++) {
                    // this will be between 0 and 1 ideally. 
                    float diff = (my_data[rowID * features_count + featureID] - clusters[clusterID][featureID]) / featureRanges[featureID];
                    dist_sq += diff * diff;
                }

                // if this is the closest cluster, classify the song as such
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    assigned_cluster = clusterID;
                }
            }
            song_assignments[rowID] = assigned_cluster;

            // add this point's data to the local sums for its assigned cluster.
            local_cluster_counts[assigned_cluster]++;
            for (int featureID = 0; featureID < features_count; featureID++) {
                local_cluster_sums[assigned_cluster * features_count + featureID] += my_data[rowID * features_count + featureID];
            }
        }

        // STEP 3 ----
        // do global sums on the cluster sums and counts
        std::vector<float> global_cluster_sums(CLUSTERS * features_count, 0.0f);
        std::vector<int> global_cluster_counts(CLUSTERS, 0);
        MPI_Allreduce(local_cluster_sums.data(), global_cluster_sums.data(), CLUSTERS * features_count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cluster_counts.data(), global_cluster_counts.data(), CLUSTERS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // update centroids with global sums/counts
        for (int c = 0; c < CLUSTERS; c++) {
            if (global_cluster_counts[c] > 0) { // avoid zero division error for empty clusters
                for (int f = 0; f < features_count; ++f) {
                    clusters[c][f] = global_cluster_sums[c * features_count + f] / global_cluster_counts[c];
                }
            }
            // empty clusters might need to be handled? unlikely?
        }
        // repeat until convergence / termination

        // reporting clusters
        if (my_rank == 0) {
            printf("cycle %d:\n", iter);
            for (int clusterID = 0; clusterID < CLUSTERS; clusterID++) {
                printf("cluster %d: ", clusterID);
                for (int featureID = 0; featureID < features_count; featureID++) {
                    printf("%f, ", clusters[clusterID][featureID]);
                }
                printf("\n");
            }
        }
    }

    // now that everything is clustered it's time for output:
    // that output should be one csv with the categorization of songs
    // and a sample visualization with 3 feature axis to show clustering.
    // so, just add a column to the csv with the assigned cluster and save.
    
    std::vector<int> all_assignments;   // all song classifications
    std::vector<int> recvcounts;        // elements expected to receive from each process
    std::vector<int> displs;            // displacements for each reception

    if (my_rank == 0) {
        all_assignments.resize(total_rows);
        recvcounts.resize(comm_sz);
        displs.resize(comm_sz);

        // calculate expected items and displacements for each thread.
        int current_displ = 0;
        for (int i = 0; i < comm_sz; ++i) {
            int rows_for_i = base_rows + (i < remainder ? 1 : 0);
            recvcounts[i] = rows_for_i;
            displs[i] = current_displ;
            current_displ += rows_for_i;
        }
    }

    // MPI_Gatherv sends my_rows # of integers from song_assignments to the root.
    // root uses recvcounts and displs to correctly place the incoming data into all_assignments.
    // using gatherv since the data sizes sent are variable.
    MPI_Gatherv(song_assignments.data(), my_rows, MPI_INT,
                all_assignments.data(), recvcounts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Successfully gathered all %d song assignments on rank 0.\n", total_rows);

        // write the original data + cluster assignments to a csv
        std::ofstream outfile("cpu-dist-results.csv");

        // original feature names (for header)
        const std::vector<std::string> feature_names = {
            "explicit", "danceability", "energy", "key", "loudness", "mode", "speechiness", 
            "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", 
            "time_signature", "year"
        };

        // write header with original features + clustering to csv
        for (const auto& name : feature_names) {
            outfile << name << ",";
        }
        outfile << "cluster_id\n"; 

        // write data to csv
        for (int r = 0; r < total_rows; r++) {
            for (int f = 0; f < features_count; f++) {
                // original features
                outfile << all_data[r * features_count + f] << ",";
            }
            // clustering assignments
            outfile << all_assignments[r] << "\n";
        }

        // should be noted that the "explicit" feature is now 0/1, not "True"/"False".

        outfile.close();
        printf("Output successfully written to cpu-dist-results.csv.\n");
    }

    MPI_Finalize();
    return 0;
}