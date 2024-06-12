# Final report
Hi!
Some small instructions for my final report.

This directory contains three implementations,
- Basic C++
- C++ and MPI
- C++ and MPI and OpenMP (very basic OpenMP usage, could probably improve it by adding some parallel sections)  

It ended up taking quite some time to get MPI working so maybe I should of done
the CUDA assignment instead... but it was fun none the less!

Instead of using the visual Python script you provided I tested my implementation
using a modified version of the reference implementation `10_cavity.py`. To test it
yourself, just run the python script (after you have built and run the C++ code!).

## Building and running
See the `Makefile`. You likely want to run the "-test" versions because they
save a matrix as .txt file.

```shell
make test_main_mpi_open_mp
mpirun -np 4 ./a.out
cd ..
python3 10_cavity.py
```

## Explanation
Also, here is a simple illustration to try explain the main idea of how the data is distributed.
![image](https://github.com/adelhult/hpsc-2024-24R51508/assets/11508459/52f706d8-7a45-4f4d-9a58-8c5bf2a66f7e)

