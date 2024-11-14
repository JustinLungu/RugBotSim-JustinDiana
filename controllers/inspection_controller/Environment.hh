#ifndef INCLUDED_ARENA_H_
#define INCLUDED_ARENA_H_

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <random>

char *wPath = getenv("WB_WORKING_DIR");

/**
 * @brief Represents an environment with a grid of tiles.
 * 
 * The Environment class manages data about the environment, including a filename,
 * a grid of tile coordinates, and a DEBUG flag for debugging.
 */
class Environment {
    std::string filename;                         ///< The name of the file containing environment data.
    std::vector<std::pair<int, int>> d_grid;      ///< The grid of tile coordinates.

public:
    int d_nrTiles = 5;       ///< The constant number of tiles.

    Environment(std::string file);
    ~Environment();

    /**
     * @brief Retrieves the color of the tile at the specified coordinates.
     * 
     * This function takes x and y coordinates as input and returns an integer value indicating the color of the tile.
     * The implementation details of determining the color are not provided in this comment.
     * 
     * @param x The x-coordinate of the tile.
     * @param y The y-coordinate of the tile.
     * @param method_read The method to read the observation: 0 for using grid data, 1 for using distributions.
     * @return An integer value representing the color of the tile (1 for white, 0 for black).
     */
    int getSample(double x, double y);

    // Set the seed for the random number generator
    void setSeed(unsigned int seed);

    // Set distributions with location information
    void setVibDistribution(double shape, double scale, double location);
    void setNonVibDistribution(double shape, double scale, double location);
    void setFPdist(int fp_prob);
    void setFNdist(int fn_prob);
    int method_read = 0;
    double vibThresh = 1.33;
    double lastSample = 0;


private:
    // Distributions with location information
    std::gamma_distribution<double> d_vibDist; // Gamma distribution for vibration
    double d_vibLoc; // Location parameter for vibration distribution

    std::gamma_distribution<double> d_nonVibDist; // Gamma distribution for non-vibration
    double d_nonVibLoc; // Location parameter for non-vibration distribution

    std::bernoulli_distribution bernoulli_FP;

    std::bernoulli_distribution bernoulli_FN;
    
    std::vector<std::vector<double>> readDataFromFile(const std::string& filename); // read accelerometer data from a file
    std::vector<std::vector<double>> data_matrix; // Matrix to store file data
    int runKeras2cppExecutable(); // run inference using keras model


    std::mt19937 d_gen_environment; // Random number generator

    /**
     * @brief Reads data from a file and populates the environment's grid.
     * 
     * This private function is called internally by the constructor to read data from a file
     * and populate the 'd_grid' member variable.
     * 
     * @throws std::ios_base::failure if the file cannot be opened.
     */
    void readFile();
};

Environment::Environment(std::string file) : filename(file) {
    // Read data from the file and populate the grid
    readFile();
}

void Environment::setFPdist(int fp_prob){
    bernoulli_FP.param(std::bernoulli_distribution::param_type(
        (double (fp_prob) / 100)
    ));
    std::cout << "Prob(FP) = " << bernoulli_FP.p() <<"\n";    
}

void Environment::setFNdist(int fn_prob){
    bernoulli_FN.param(std::bernoulli_distribution::param_type(
        (double (fn_prob) / 100)
    ));
    std::cout << "Prob(FN) = " << bernoulli_FN.p() <<"\n";  
}


void Environment::readFile() {
    std::ifstream file;
    if (wPath != NULL) {
        char file_name[256];
        std::cout << "Webots working dir enabled" << '\n';
        sprintf(file_name, "%s/world.txt", wPath);
        file.open(file_name);
    } else {
        file.open(filename);
    }

    if (!file.is_open())
        throw std::ios_base::failure("The file cannot be read");

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        int x, y;
        char comma;

        if (lineStream >> x >> comma >> y) {
            d_grid.push_back(std::make_pair(x, y));
            std::cout << x << y << "\n";
        } else {
            std::cout << "Unsupported file formatting in world file" << '\n';
        }
    }
    std::cout << std::endl;
    file.close(); // Close the file
}

void Environment::setSeed(unsigned int seed) {
    d_gen_environment.seed(seed);
    std::cout<<"environmental seed: " <<seed<<'\n';
}

void Environment::setVibDistribution(double shape, double scale, double location) {
    d_vibDist = std::gamma_distribution<double>(shape, scale);
    d_vibLoc = location;
}

void Environment::setNonVibDistribution(double shape, double scale, double location) {
    d_nonVibDist = std::gamma_distribution<double>(shape, scale);
    d_nonVibLoc = location;
}


int Environment::getSample(double x, double y) {
    if (method_read == 0) { // Use grid data
        for (std::pair<int, int> coloredTile : d_grid) {
            if (
                (x >= 1.0 * coloredTile.first / d_nrTiles) &&
                (x <= (1.0 * coloredTile.first + 1) / d_nrTiles) &&
                (y >= 1.0 * coloredTile.second / d_nrTiles) &&
                (y <= (1.0 * coloredTile.second + 1) / d_nrTiles)
            ) {
                lastSample = 1;

                return 1; // WHITE TILE
            }
        }
        lastSample = 0;
        return 0; // BLACK TILE
    }

    if (method_read == 2) { // Use grid data and FP / FN
        for (std::pair<int, int> coloredTile : d_grid) {
            if (
                (x >= 1.0 * coloredTile.first / d_nrTiles) &&
                (x <= (1.0 * coloredTile.first + 1) / d_nrTiles) &&
                (y >= 1.0 * coloredTile.second / d_nrTiles) &&
                (y <= (1.0 * coloredTile.second + 1) / d_nrTiles)
            ) {
                // Sample false negative (FN) with probability `fn_prob`
                if (bernoulli_FN(d_gen_environment)) {
                    lastSample = 0; // False negative: white classified as black
                    return 0; // BLACK TILE due to FN
                } else {
                    lastSample = 1; // True positive: white classified as white
                    return 1; // WHITE TILE
                }
            }
        }
        if (bernoulli_FP(d_gen_environment)) {
            lastSample = 1; // False positive: black classified as white
            return 1; // WHITE TILE due to FP
        } else {
            lastSample = 0; // True negative: black classified as black
            return 0; // BLACK TILE
        }
    }

    if (method_read ==3){ // Use Neural Network to inference on accelerometer data
        for (std::pair<int, int> coloredTile : d_grid) {
            if (
                (x >= 1.0 * coloredTile.first / d_nrTiles) &&
                (x <= (1.0 * coloredTile.first + 1) / d_nrTiles) &&
                (y >= 1.0 * coloredTile.second / d_nrTiles) &&
                (y <= (1.0 * coloredTile.second + 1) / d_nrTiles)
            ) {
                // lastSample = 1; 
                data_matrix = readDataFromFile("../../data/capture2_40hz_60vol.txt"); // white tile, 1
            }
        }
        // lastSample = 0;
        data_matrix = readDataFromFile("../../data/capture1_60hz_30vol.txt"); // black tile, 0


        // Writing current data in a file to be read for inference
        std::ofstream file("../keras2cpp/build/curr_data.txt", std::ofstream::trunc);  

        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << std::endl;
        }

        for (const auto& row : data_matrix) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) file << "\t";  
            }
            file << "\n";  
        }
        
        file.close();

        // Call the keras2cpp executable and capture its output
        int result = runKeras2cppExecutable();
        if (result == -1) {
            std::cout << "Something went very wrong... \n";
        }
        if (result == 2) {
            lastSample = 1;
            return 1;
        } else {
            if (result == 0) {
                lastSample = 0;
                return 0;
            }
        }  
    }

    if (method_read == 1) { // Use distributions
        for (std::pair<int, int> coloredTile : d_grid) {
            
            if (
                (x >= 1.0 * coloredTile.first / d_nrTiles) &&
                (x <= (1.0 * coloredTile.first + 1) / d_nrTiles) &&
                (y >= 1.0 * coloredTile.second / d_nrTiles) &&
                (y <= (1.0 * coloredTile.second + 1) / d_nrTiles)
            ) {
                // Sample from vibration distribution and apply location shift
                lastSample = d_vibDist(d_gen_environment) + d_vibLoc;
                if (lastSample > vibThresh) {return 1;}
                else {return 0;}
            }
        }
        // Sample from non-vibration distribution and apply location shift
        lastSample = d_nonVibDist(d_gen_environment) + d_nonVibLoc;
        if (lastSample > vibThresh) {return 1;}
        else {return 0;}
    }

    return 0; // Exception case
}


///// Additional functions for NN implementation:

std::vector<std::vector<double>> Environment::readDataFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data_matrix;

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return data_matrix; // Return an empty matrix if file cannot be opened
    }

    // choose a random point in the file to sample from
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 36000); // Ensure we can read 24 rows
    int start_row = dis(gen);

    
    std::string line;
    int current_row = 0;
    while (std::getline(file, line)) {
        // If the current row is less than the start row, skip it
        if (current_row < start_row) {
            ++current_row;
            continue;
        }

        // Stop reading if we have read 24 rows
        if (data_matrix.size() >= 24) {
            break;
        }

        // Parse the line into doubles and store it in the matrix
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        while (iss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            data_matrix.push_back(row); // Add the row to the matrix
        }

        ++current_row;
    }

    file.close();

    return data_matrix; // Return the matrix
}


int Environment::runKeras2cppExecutable() {
    // Save the current working directory
    char originalDir[256];
    getcwd(originalDir, sizeof(originalDir));

    // Change to the keras2cpp/build directory
    chdir("../keras2cpp/build");

    // Run keras2cpp with the dynamically generated data file as input
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("./keras2cpp curr_data.txt", "r"), pclose);
    if (!pipe) {
        std::cerr << "Error: popen() failed" << std::endl;
        return -1;
    }

    // Capture the output from keras2cpp line by line
    char buffer[128];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        output += buffer;
    }

    // Restore the original working directory
    chdir(originalDir);

    // Remove square brackets and replace commas with spaces
    output.erase(std::remove_if(output.begin(), output.end(),
                                [](char c) { return c == '[' || c == ']'; }),
                 output.end());
    std::replace(output.begin(), output.end(), ',', ' ');

    // Print the cleaned output
    std::cout << "Cleaned keras2cpp output: " << output << std::endl;

    // Parse output to find the index with the highest value
    std::istringstream iss(output);
    std::vector<double> values;
    double number;
    while (iss >> number) {
        values.push_back(number);
    }

    // Find the index of the maximum value
    if (!values.empty()) {
        auto max_it = std::max_element(values.begin(), values.end());
        int max_index = std::distance(values.begin(), max_it);
        return max_index;
    } else {
        std::cout << "No values found in the output." << std::endl;
        return -1;
    }
}



// Destructor definition
Environment::~Environment() {
    // Default destructor generated by the compiler handles cleanup
}

#endif
