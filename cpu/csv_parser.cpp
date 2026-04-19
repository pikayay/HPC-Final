#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

// Function to safely parse a CSV line respecting double quotes
vector<string> parseCSVLine(const string& line) {
    vector<string> result;
    string current = "";
    bool in_quotes = false;
    
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            result.push_back(current);
            current = "";
        } else {
            current += c;
        }
    }
    result.push_back(current);
    return result;
}

int main() {
    // Make sure this matches the name of your raw Kaggle download
    string input_filename = "tracks_features.csv"; 
    string output_filename = "tracks_features_cleaned.csv";

    ifstream infile(input_filename);
    ofstream outfile(output_filename);

    if (!infile.is_open()) {
        cerr << "Error: Could not open input file: " << input_filename << endl;
        return 1;
    }
    if (!outfile.is_open()) {
        cerr << "Error: Could not open output file: " << output_filename << endl;
        return 1;
    }

    string line;
    int row_count = 0;

    // Read and skip the original header row
    if (getline(infile, line)) {
        // Write a new header including the id and the 15 features
        outfile << "id,explicit,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature,year\n";
    }

    cout << "Parsing data..." << endl;

    while (getline(infile, line)) {
        vector<string> columns = parseCSVLine(line);
        
        // Ensure the row parsed correctly and has enough columns
        if (columns.size() >= 23) { 
            
            // Write the ID (index 0) followed by a comma
            outfile << columns[0] << ",";

            // Write only indices 8 through 22
            for (int i = 8; i <= 22; ++i) {
                outfile << columns[i];
                if (i < 22) {
                    outfile << ",";
                }
            }
            outfile << "\n";
            row_count++;
            
            // Print progress every 100,000 rows
            if (row_count % 100000 == 0) {
                cout << "Processed " << row_count << " rows..." << endl;
            }
        }
    }

    infile.close();
    outfile.close();

    cout << "Finished! Cleaned dataset saved to " << output_filename << endl;
    cout << "Total valid rows exported: " << row_count << endl;

    return 0;
}