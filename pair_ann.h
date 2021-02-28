/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
 
 Pair style ann contributed by Chun-Wei Pao at Academia Sinica, Taiwan
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(ann,PairAnn)

#else
//test
#ifndef LMP_PAIR_ANN_H
#define LMP_PAIR_ANN_H

#include "pair.h"

namespace LAMMPS_NS {

class PairAnn : public Pair {
 public:
  PairAnn(class LAMMPS *);
  virtual ~PairAnn();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  double init_one(int, int);

 protected:

  

// structure of finger print G2
  struct G2FingerPrint{
      int NG2;  // number of G2 finger print of given atom type
      int *T2;  // array of the 2nd atom type with size equal to NG2 
      double *eta;  // array of zeta with size equal to NG2
      double *minima; // 
      double *maxima;
  };
  
  G2FingerPrint *G2; // G2[total no. of atom types]

// structure of finger print G4
  struct G4FingerPrint{
      int NG4; 
      int *G4T2;
      int *G4T3;
      double *G4eta;
      double *G4gamma;  //actually lambda
      double *G4zeta;
      double *G4minima;
      double *G4maxima;
  };
  
  G4FingerPrint *G4; // G4[total no. of atom types]
  int *NoFingerPrints; // total number of finger prints of each element 


  struct HiddenLayer{
    int Nnodeminus1;  // number of nodes of the previous layer
    int Nnode;  // number of nodes of current layer
    double **weight;  // weight matrix
  };

  struct OutLayer{
    int NnodeOut; // number of nodes in the last layer
    double *outweight; // weight of the output layer
  };

  int activation; // activation function of each element
                   // 1 'tanh'

  HiddenLayer *Layer; // Layer[0-n] => hidden layers of element 1 with n-1 hidden layers
  int *NoLayers;              // number of layers of each elements

  OutLayer *OutLay; 

  double *Linear; // coefficient for final summation of neural network
  double *Intercept; // coefficient for final summation of neural network

  double **rij;
  double ***Rij;
  double ***uij;
  double **fc;
  double **dfc;
  double **expr2;

  double **fingerprint; // fingerprint of each atoms
  double **fingerprintprime; // derivative of fingerprints

  int MaxNoFingerprints; // the max no of finger print for all elements
  int MaxNoLayers; // max number of hidden layers
  int MaxNoNodes; // the max number of nodes in the hidden layer
  int TotalLayer; // total number of hidden layers
  char **elements;              // names of unique elements
  int *map;                     // mapping from atom types to elements
  double cutmax;                // max cutoff for all elements
  int nelements;                // # of unique elements
  int maxshort;                 // size of short neighbor list array
  int **neighshort;              // short neighbor list array
  int maxlocal;

  int fp_spec_num;      // show specfic # of fingerprint for all atoms
          
  virtual void allocate();
  virtual void read_file(char *);
  virtual double compute_G2(double,double,double);
  virtual double fast_compute_G2(double,double,double);

  virtual double compute_G4(double,double,double,double,double,double,double *,double *,double);
  virtual double fast_compute_G4(double , double , double , double , double , double );
  virtual double fast_fast_compute_G4(double , double , double , double , double );
  
  virtual double Projector2(int ,int );
  virtual double compute_G2_prime(int ,int ,int ,int ,int ,int ,int ,double ,double ,double ***,double );
  virtual double compute_dF2dqu(int ,int ,int ,double ,double ,double ***,double ); 
  virtual double fast_compute_dF2dqu(double ,double ,double ,double ,double ,double ,double );

  virtual double filter(int ,int ,int ,int );
  virtual double compute_G4_prime(int ,int ,int ,int ,int ,int ,int ,int ,int ,double ,double ,double ,double ,double ,double ,double ***,double );

  virtual double compute_dAF4dqu(int ,int ,int ,int ,double ,double ,double ,double ,double ,double ,double ***,double );
  virtual double fast_compute_dAF4dqu(int ,int ,int ,int ,double ,double ,double ,double ,double ,double ,double ,double ,double ,double ***,double );
  virtual double fast_fast_compute_dAF4dqu(int ,int ,int ,int ,double ,double ,double ,double ,double ,double ,double ,double ,double ***,double );

  virtual double j_compute_dAF4dqu(int ,int ,int ,int ,double ,double ,double ,double ,double ,double ,double ***,double );
  virtual double fast_j_compute_dAF4dqu(int ,int ,int ,int ,double ,double ,double ,double **,double **,double ,double **,double **,double ,double ***,double );
  virtual double fast_fast_j_compute_dAF4dqu(int ,int ,int ,int ,double ,double ,double ,double **,double **,double ,double **,double **,double ***,double );
//  virtual double common_fast_fast_compute_dAF4dqu(double *,int ,int ,int ,int ,double ,double ,double ,double **,double **,double ,double **,double **,double ***,double );
  virtual void common_fast_fast_compute_dAF4dqu(double *,int ,int ,int ,int ,double ,double ,double ,double **,double **,double ,double **,double **,double ***,double );
 // inlined functions for efficiency
  
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.


E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.



*/
