#give execution acces
#chmod +x ../compile.sh
#./compile.sh

export WEBOTS_HOME=/usr/local/webots

#!/bin/bash
cd "$(dirname "$0")"

# Clean up CMake-generated files in keras2cpp build directory
echo "Cleaning up CMake files in keras2cpp build directory..."
cd keras2cpp/build
rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake Makefile temp_input.txt

# Compile keras2cpp
echo "Configuring and compiling keras2cpp..."
cmake ..  # Configure every time
cmake --build . --clean-first  # Build keras2cpp

cd ../../

cd inspection_controller
make clean
make

cd ../

cd cpp_supervisor
make clean
make

cd ../