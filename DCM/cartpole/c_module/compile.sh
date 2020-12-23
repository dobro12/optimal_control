g++ -fPIC -c main.cpp -I/usr/local/include/python3.6m -I/usr/include/python3.6m
g++ -shared -o main.so main.o
