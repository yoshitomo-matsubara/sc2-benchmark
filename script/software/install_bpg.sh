wget https://bellard.org/bpg/libbpg-0.9.8.tar.gz
tar -xvf libbpg-0.9.8.tar.gz
cd libbpg-0.9.8/
sudo apt-get -y install libpng-dev
sudo apt-get -y install libjpeg-dev
sudo apt-get -y install libsdl-dev
sudo apt-get -y install libsdl-image1.2-dev
sudo apt-get remove libnuma-dev
sudo make
sudo apt-get install libnuma-dev
