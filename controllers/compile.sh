#give execution acces
#chmod +x ../compile.sh
#./compile.sh

export WEBOTS_HOME=/usr/local/webots

#!/bin/bash
cd "$(dirname "$0")"

# Compile keras2cpp
echo "Configuring and compiling keras2cpp..."
cd keras2cpp/build
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