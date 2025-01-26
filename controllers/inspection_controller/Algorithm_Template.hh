#ifndef INCLUDED_ALGORITHM1_HH_
#define INCLUDED_ALGORITHM1_HH_
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <utility>
#include <filesystem>
#include <sstream>

#include <unistd.h> // for chdir
#include <cstdio>
#include <memory>
#include <thread>
#include <chrono>
#include <string>
#include <algorithm>
#include <cctype>

#include "RugBot.hh"
#include "Environment.hh"
#include "radio.hh"
#include "controller_settings.hh"

typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;
typedef std::vector<double> Array;

class Algorithm1 {
public:
    enum AlgoStates {
        STATE_RW,  // State: Random Walk
        STATE_OBS, // State: Observe Color
        STATE_PAUSE // etc....
    };

    AlgoStates states = STATE_RW;

    // Time step for the simulation
    enum { TIME_STEP = 20 };

    Algorithm1() : settings(), robot(TIME_STEP), environ("world.txt"), radio(robot.d_robot, TIME_STEP) {};

    void run();
    void recvSample();
    void sendSample(int sample);
    void runKeras2cppExecutable(const std::vector<double>& inputValues);

private:
    ControllerSettings settings;
    RugRobot robot;
    Environment environ;
    Radio_Rover radio;
    std::vector<int> pos;
    int sample_ = 0;
    std::vector<std::vector<double>> data_matrix; // Matrix to store file data
    std::vector<std::vector<double>> readDataFromFile(const std::string& filename);
    std::vector<double> inputValues;
    std::vector<int> robotPos;
};

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdio>
#include <cstring>
#include <unistd.h>

void Algorithm1::runKeras2cppExecutable(const std::vector<double>& inputValues) {
    // Ensure the input vector has the correct size
    if (inputValues.size() != 3) {
        std::cerr << "Error: Expected exactly 3 input values (shape 3,)." << std::endl;
        return;
    }

    // Save the current working directory
    char originalDir[256];
    if (!getcwd(originalDir, sizeof(originalDir))) {
        std::cerr << "Error: Unable to get current working directory." << std::endl;
        return;
    }

    // Change to the keras2cpp/build directory
    if (chdir("../keras2cpp/build") != 0) {
        std::cerr << "Error: Unable to change to keras2cpp/build directory." << std::endl;
        return;
    }

    // Create a temporary input file with the values
    const char* tempFileName = "temp_input.txt";
    std::ofstream tempFile(tempFileName);
    if (!tempFile.is_open()) {
        std::cerr << "Error: Unable to create temporary input file." << std::endl;
        // Restore original directory before returning
        chdir(originalDir);
        return;
    }

    // Write the input values to the temporary file
    for (const double& value : inputValues) {
        tempFile << value << " ";
    }
    tempFile.close();

    // Run keras2cpp with the temporary input file
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("./keras2cpp temp_input.txt", "r"), pclose);
    if (!pipe) {
        std::cerr << "Error: popen() failed." << std::endl;
        // Restore original directory before returning
        chdir(originalDir);
        return;
    } else {
        std::cout << "popen() succeeded. Running keras2cpp..." << std::endl;
    }

    // Read the output from the executable
    char buffer[128];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        output += buffer;
    }

    // Parse and process the output
    std::cout << "Model output from alg_template: " << output << std::endl;

    // Restore the original working directory
    if (chdir(originalDir) != 0) {
        std::cerr << "Error: Unable to restore the original working directory." << std::endl;
    }
}


void Algorithm1::run() {
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    robot.setCustomData("");
    settings.readSettings();

    while (robot.d_robot->step(TIME_STEP) != -1) {
        switch (states) {
            case STATE_RW:
                if (robot.RandomWalk() == 1) {
                    states = STATE_OBS;
                }
                break;

            case STATE_OBS:
                sample_ = environ.getSample(robot.getPos()[0], robot.getPos()[1]);
                std::cout << "Sample: " << sample_ << " Robot position: " 
                          << 1.0 * robot.getPosXYZ()[0] / 100 << " " << 1.0 * robot.getPosXYZ()[1] / 100 << " " << 1.0 * robot.getPosXYZ()[2] / 100 << "\n";
                // Get the robot's XYZ position
                robotPos = robot.getPosXYZ();

                // Convert the XYZ coordinates to the desired scale for inputValues
                inputValues = {
                    1.0 * robotPos[0] / 100,  // X coordinate scaled
                    1.0 * robotPos[1] / 100,  // Y coordinate scaled
                    1.0 * robotPos[2] / 100   // Z coordinate scaled
                };

                // Use robot position to run inference on model:

                runKeras2cppExecutable(inputValues);
            

                states = STATE_RW;
                break;

            case STATE_PAUSE:
                // Pause logic here
                break;
        }
        
        if (robot.d_robot->getTime() > 5) {
            robot.setCustomData(std::to_string((int)robot.d_robot->getTime()) +
                "," + std::string(robot.d_robot->getName().substr(1, 1)));
        }
    }
}


std::vector<std::vector<double>> Algorithm1::readDataFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data_matrix;

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return data_matrix; // Return an empty matrix if file cannot be opened
    }

    // Get the current timestamp as the starting row number
    int start_row = static_cast<int>(robot.d_robot->getTime()) - 24;
    
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


void Algorithm1::recvSample() {
    std::vector<int> messages = radio.getMessages();
    for (int sample : messages) {
        // Process received messages
    }
}

void Algorithm1::sendSample(int sample) {
    int const *message;
    message = &sample;
    radio.sendMessage(message, sizeof(message));
}

#endif // INCLUDED_ALGORITHM1_HH_
