mkdir ~/software
cd ~/software
git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
cd VVCSoftware_VTM
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
