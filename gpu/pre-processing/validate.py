import csv
import sys

def get_clusters(filename):
    clusters = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) > 1:
                song_id = row[0]
                cluster_id = row[-1]
                clusters[song_id] = cluster_id
    return clusters

def main():
    if len(sys.argv) != 3:
        print("Usage: python validate.py serial_results.csv parallel_results.csv")
        return

    serial_data = get_clusters(sys.argv[1])
    parallel_data = get_clusters(sys.argv[2])

    if len(serial_data) != len(parallel_data):
        print("Validation Failed: Row counts do not match")
        return

    total_rows = len(serial_data)
    mismatches = 0
    
    for song_id in serial_data:
        if song_id not in parallel_data:
            print("Missing ID: " + song_id)
            mismatches += 1
        elif serial_data[song_id] != parallel_data[song_id]:
            mismatches += 1

    # Calculate the percentage of mismatches
    error_rate = (mismatches / total_rows) * 100
    tolerance = 0.01 # 0.01% acceptable variance for floating point non-associativity

    if mismatches == 0:
        print("Validation passed. All cluster assignments match perfectly.")
    elif error_rate <= tolerance:
        print(f"Validation passed with acceptable floating-point variance.")
        print(f"Mismatches: {mismatches} out of {total_rows} ({error_rate:.5f}%)")
        print("Note: Minor deviations are expected due to parallel reduction non-associativity.")
    else:
        print(f"Validation failed. Mismatch rate ({error_rate:.5f}%) exceeds tolerance.")
        print(f"Total mismatches: {mismatches}")

if __name__ == "__main__":
    main()