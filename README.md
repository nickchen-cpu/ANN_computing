# pair_ann


## Introduction
We use lammps to calculate the energy of a molecular structure.
![image](https://user-images.githubusercontent.com/70849044/109465821-bc291900-7aa3-11eb-8ace-4b194ca6fb86.png)









## Build
**Suppose that you are in current WORK_DIR**
```
1. git clone -b master https://github.com/lammps/lammps.git mylammps
2. cp src/pair_ann.{h,cpp}  mylammps/src/
3. cd mylammps              # change to the LAMMPS distribution directory
4. mkdir build; cd build    # create and use a build directory
5. cmake ../cmake           # configuration reading CMake scripts from ../cmake
6. cmake --build .          # compilation (or type "make")
7. mv ../src/lmp_mpi ${WORK_DIR}
```
## Usage
```
1. cd  ${WORK_DIR}
2. mpirun -np 4 lmp_mpi -v number 0 -in montecarlo_dynamic.in
```
