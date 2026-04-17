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


int main(void) {
    int comm_sz; // threadcount
    int my_rank; // rank
    srand(42);
    const int CLUSTERS = 8;
    const int features_count = 15; 
    const int MAX_ITERATIONS = 100; // match gpu default
    const float tol = 1e-4f; // convergence tolerance

    // vars for distributing data
    int total_rows = 0;
    std::vector<float> all_data; // big vector, populated on p0
    std::vector<std::string> ids;     // populated on p0 so we don't gotta load the csv twice
    float clusters[CLUSTERS][features_count]; // shared centroids

    // initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    // process 0 reads file into memory
    if (my_rank == 0) {
        rapidcsv::Document source("tracks_features_cleaned.csv");
        total_rows = source.GetRowCount();
        
        // Load IDs separately. They are only needed on rank 0 for the final output.
        ids = source.GetColumn<std::string>("id");

        // features extracted for clustering, in order
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
        
        // normalize features to 0-1 range.
        // begin by getting the minimum and maximum of every feature
        std::vector<float> fmin(features_count, std::numeric_limits<float>::max());
        std::vector<float> fmax(features_count, -std::numeric_limits<float>::max());
        // for every song:
        for (int i = 0; i < total_rows; ++i) {
            // for every feature:
            for (int d = 0; d < features_count; ++d) {
                // get the min and max
                float val = all_data[i * features_count + d];
                if (val < fmin[d]) fmin[d] = val;
                if (val > fmax[d]) fmax[d] = val;
            }
        }
        // now establish ranges for each feature and normalize the entries based on that
        for (int i = 0; i < total_rows; ++i) {
            for (int d = 0; d < features_count; ++d) {
                float range = fmax[d] - fmin[d];
                all_data[i * features_count + d] = (range > 0) ? (all_data[i * features_count + d] - fmin[d]) / range : 0.0f;
            }
        }

        // initialize centroids by picking a random song
        std::vector<int> indices;
        while ((int)indices.size() < CLUSTERS) {
            int idx = rand() % total_rows;
            bool dup = false;
            for (int x : indices) if (x == idx) { dup = true; break; }
            if (!dup) indices.push_back(idx);
        }

        for (int c = 0; c < CLUSTERS; ++c) {
            int row = indices[c];
            for (int d = 0; d < features_count; ++d) {
                clusters[c][d] = all_data[row * features_count + d];
            }
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

    // gonna distribute data from thread 0 to all threads using scatterv
    // so here's the necessary arrays:
    std::vector<int> scatter_sendcounts;
    std::vector<int> scatter_displs;

    if (my_rank == 0) {
        scatter_sendcounts.resize(comm_sz);
        scatter_displs.resize(comm_sz);
        int current_displ = 0;
        // calculate send counts and displacements for the data scattering
        for (int i = 0; i < comm_sz; i++) {
            int rows_for_i = base_rows + (i < remainder ? 1 : 0);
            scatter_sendcounts[i] = rows_for_i * features_count; // # of floats i gets
            scatter_displs[i] = current_displ;
            current_displ += scatter_sendcounts[i]; // skip forward that # of floats for next displacement
        }
    }

    // scatter everything out from p0 to all threads
    MPI_Scatterv(all_data.data(), scatter_sendcounts.data(), scatter_displs.data(), MPI_FLOAT,
                 my_data.data(), my_rows * features_count, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    if (my_rank == 0) printf("Data distributed to all processes via Scatterv.\n");

    double start_time = 0;
    if (my_rank == 0) start_time = MPI_Wtime();
    bool converged = false;

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
                    // data already normalized so no worries there
                    float diff = my_data[rowID * features_count + featureID] - clusters[clusterID][featureID];
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

        // store old centroids to check for convergence
        float old_clusters[CLUSTERS][features_count];
        memcpy(old_clusters, clusters, CLUSTERS * features_count * sizeof(float));

        // update centroids with global sums/counts
        // this is done on all processes, so they all have the updated centroids
        for (int c = 0; c < CLUSTERS; c++) {
            if (global_cluster_counts[c] > 0) { // avoid zero division error for empty clusters
                for (int f = 0; f < features_count; ++f) {
                    clusters[c][f] = global_cluster_sums[c * features_count + f] / global_cluster_counts[c];
                }
            }
            // empty clusters keep their old centroid values (handled by memcpy earlier)
        }

        // convergence check
        float max_move = 0.0f;
        for (int c = 0; c < CLUSTERS; c++) {
            for (int f = 0; f < features_count; ++f) {
                float move = std::fabs(clusters[c][f] - old_clusters[c][f]);
                if (move > max_move) {
                    max_move = move;
                }
            }
        }

        // report changes
        if (my_rank == 0) {
            printf("Iteration %d, max centroid movement: %f\n", iter, max_move);
        }
        if (max_move < tol) {
            converged = true;
            break; // all processes break synchronously
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
        double end_time = MPI_Wtime();
        printf("K-means finished%s in %.4f s\n",
               (converged ? " (converged)" : " (max iterations)"),
               end_time - start_time);
    }

    if (my_rank == 0) {
        printf("Successfully gathered all %d song assignments on rank 0.\n", total_rows);

        // write the original data + cluster assignments to a csv
        std::ofstream outfile("cpu-dist-results.csv");

        // feature names for output csv header
        const std::vector<std::string> feature_names = {
            "id", "explicit", "danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", 
            "time_signature", "year"
        };

        // write header with original features + clustering to csv
        for (const auto& name : feature_names) {
            outfile << name << ",";
        }
        outfile << "cluster\n"; 

        // write data to csv
        for (int r = 0; r < total_rows; r++) {
            // write IDs from ids stored during data load
            outfile << ids[r] << ",";
            for (int f = 0; f < features_count; f++) {
                // original features
                outfile << all_data[r * features_count + f] << ",";
            }
            // clustering assignments
            outfile << all_assignments[r] << "\n";
        }

        outfile.close();
        printf("Output successfully written to cpu-dist-results.csv.\n");
    }

    MPI_Finalize();
    return 0;
}