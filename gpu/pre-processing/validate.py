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
        print("Usage: python validate.py serial_results.csv gpu_results.csv")
        return

    serial_data = get_clusters(sys.argv[1])
    gpu_data = get_clusters(sys.argv[2])

    if len(serial_data) != len(gpu_data):
        print("Row counts do not match")
        return

    mismatches = 0
    for song_id in serial_data:
        if song_id not in gpu_data:
            print("Missing ID: " + song_id)
            mismatches += 1
        elif serial_data[song_id] != gpu_data[song_id]:
            mismatches += 1

    if mismatches == 0:
        print("Validation passed. All cluster assignments match perfectly.")
    else:
        print("Validation failed with " + str(mismatches) + " mismatches.")

if __name__ == "__main__":
    main()