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


    int method_read = 0;
    double vibThresh = 1.33;
    double lastSample = 0;


private:



    



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



void Environment::readFile() {
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    
    std::ifstream file;
    if (wPath != NULL) {
        char file_name[256];
        std::cout << "Webots working dir enabled" << '\n';
        std::cout << filename << '\n';
        sprintf(file_name, "%s/world.txt", wPath);
        std::cout << file_name << '\n';
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





int Environment::getSample(double x, double y) {

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

// Destructor definition
Environment::~Environment() {
    // Default destructor generated by the compiler handles cleanup
}

#endif
