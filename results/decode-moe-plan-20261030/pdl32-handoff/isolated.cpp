#include <pybind11/pybind11.h>
#include <stdint.h>
int d32o_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int d32o_attr(int);
int d32u_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);
int d32u_info(int);
int d32u_occupancy();
