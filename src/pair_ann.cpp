/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Chun-Wei Pao, Academia Sinica, Taiwan
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_ann.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include <fstream>
#include <iostream>
#include <cstring>

#include "math_const.h"
// test
using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairAnn::PairAnn(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  G2 = NULL;
  G4 = NULL;

  activation = 0;

  rij  = NULL;
  Rij  = NULL;
  uij = NULL;
  fc  = NULL;
  dfc  = NULL;
  expr2 = NULL;

  fingerprint = NULL;
  fingerprintprime = NULL;

  nelements = 0;
  elements = NULL;
  Layer = NULL;
  map = NULL;
  NoLayers = NULL;
  NoFingerPrints = NULL;
  Linear = NULL;
  Intercept = NULL;
  OutLay = NULL;
  MaxNoFingerprints = 0;
  MaxNoNodes = 50;
  MaxNoLayers = 50;
 
  maxlocal = 0;
//  maxshort = 50;
  maxshort = 200;
  neighshort = NULL;
  fp_spec_num = 0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairAnn::~PairAnn()
{
  if (copymode) return;

  if (elements){
    for (int i = 0; i < nelements; i++){
      delete [] elements[i];
    //   delete [] NoFingerPrints[i];
    //   delete [] NoLayers[i];
    }
    delete [] elements;
    delete [] NoFingerPrints;
    delete [] NoLayers;
  }

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
//    memory->destroy(neighshort);
    delete [] map;
  }

  if (G2) {
    for (int i = 0; i < nelements; i++) {
      //delete [] G2[i].file;
      memory->destroy(G2[i].T2);
      memory->destroy(G2[i].eta);
      memory->destroy(G2[i].minima);
      memory->destroy(G2[i].maxima);
    }
    memory->sfree(G2);
    G2 = NULL;
  }


  if (G4) {
    for (int i = 0; i < nelements; i++) {
      //delete [] G2[i].file;
      memory->destroy(G4[i].G4T2);
      memory->destroy(G4[i].G4T3);
      memory->destroy(G4[i].G4eta);
      memory->destroy(G4[i].G4gamma);
      memory->destroy(G4[i].G4zeta);
      memory->destroy(G4[i].G4minima);
      memory->destroy(G4[i].G4maxima);
    }
    memory->sfree(G4);
    G4 = NULL;
  }


  if(OutLay) {
    for (int i =0; i < nelements; i++) {
      memory->destroy(OutLay[i].outweight);
    }
    memory->sfree(OutLay);
    OutLay = NULL;
  }

  if (Layer) {
    for (int i = 0; i < TotalLayer; i++) {
      //delete [] G2[i].file;
      memory->destroy(Layer[i].weight);
    }
    memory->sfree(Layer);
    Layer = NULL;
  }
}

/* ---------------------------------------------------------------------- */

void PairAnn::compute(int eflag, int vflag)
{
    int i,j,k,ii,jj,kk,nn,mm;

    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = vflag_fdotr = vflag_atom = 0;

    // initialize an array to store values in each nodes in the neural network
    if (atom->nmax > maxlocal) {
	maxlocal = atom->nmax;
	memory->destroy(fingerprint);
	memory->destroy(fingerprintprime);
	memory->create(fingerprint,maxlocal,MaxNoFingerprints,"ANN:fingerprint");
	memory->create(fingerprintprime,maxlocal,MaxNoFingerprints,"ANN:fingerprintprime");

	memory->destroy(rij);
	memory->create(rij,2000,2000,"ANN:rij");
//	memory->create(rij,maxlocal,maxlocal,"ANN:rij");
	memory->destroy(Rij);
	memory->create(Rij,2000,2000,3,"ANN:Rij");
//	memory->create(Rij,maxlocal,maxlocal,3,"ANN:Rij");

	memory->destroy(uij);
	memory->create(uij,2000,2000,3,"ANN:uij");

	memory->destroy(fc);
	memory->create(fc,2000,2000,"ANN:fc");
	memory->destroy(dfc);
	memory->create(dfc,2000,2000,"ANN:dfc");
	memory->destroy(expr2);
	memory->create(expr2,2000,2000,"ANN:expr2");
	memory->destroy(neighshort);
	memory->create(neighshort,maxlocal,maxshort,"ANN:neighshort");
    }

    double **x = atom->x;
    double **f = atom->f;
    tagint *tag = atom->tag;
    int *type = atom->type;

    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    int inum,*ilist,*numneigh,**firstneigh;
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    int jnum,*jlist;

    int *num;
    num=(int *)malloc(sizeof(int)*inum);
    int numshort[2000];
    for (ii = 0; ii < 2000; ii++) numshort[ii]=0;

    int u;
    double r,rc,r2,d[3];
    rc=cutmax;

    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	jlist = firstneigh[i];
	jnum = numneigh[i];

	for (jj = 0; jj < jnum; jj++) {
	    j = jlist[jj];
	    j &= NEIGHMASK;
	    if(j<=i) continue;

	    for(u=0;u<3;u++) d[u]=x[i][u]-x[j][u];
	    
	    r2=0.0;
	    for(u=0;u<3;u++) r2=r2+d[u]*d[u];
	    r=sqrt(r2);

	    if (r > rc) continue;

	    neighshort[i][numshort[i]++] = j;
	    neighshort[j][numshort[j]++] = i;

	    for(u=0;u<3;u++) {
		Rij[i][j][u]=d[u];
		uij[i][j][u]=d[u]/r;

		Rij[j][i][u]=-d[u];
		uij[j][i][u]=-d[u]/r;
	    }

	    rij[i][j]=r;
	    rij[j][i]=r;

	    fc[i][j]=(1.0+cos(MY_PI*r/rc))/2.0;
	    fc[j][i]=fc[i][j];

	    dfc[i][j]=-(MY_PI/rc*sin(MY_PI*r/rc)/2.0);
	    dfc[j][i]=dfc[i][j];
	    
	    expr2[i][j]=exp(r2/rc/rc);
	    expr2[j][i]=expr2[i][j];
	    
	}
	*(num+i)=numshort[i];
    }

    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    for (kk = 0; kk < *(num+i); kk++) {
		k = neighshort[i][kk];
		k &= NEIGHMASK;
		if(k<=j) continue;
		for(u=0;u<3;u++) d[u]=Rij[k][i][u]-Rij[j][i][u];

		r2=0.0;
		for(u=0;u<3;u++) r2=r2+d[u]*d[u];
		r=sqrt(r2);

		rij[k][j]=r;
		rij[j][k]=r;

		if(r>rc) continue;

		for(u=0;u<3;u++) {
		    Rij[k][j][u]=d[u];
		    uij[k][j][u]=d[u]/r;

		    Rij[j][k][u]=-d[u];
		    uij[j][k][u]=-d[u]/r;
		}

		fc[k][j]=(1.0+cos(MY_PI*r/rc))/2.0;
		fc[j][k]=fc[k][j];

		dfc[k][j]=-(MY_PI/rc*sin(MY_PI*r/rc)/2.0);
		dfc[j][k]=dfc[k][j];

		expr2[k][j]=exp(r2/rc/rc);
		expr2[j][k]=expr2[k][j];
	    }
	}
    }


    double ***fcfcfc;
    fcfcfc=(double ***)malloc(sizeof(double **)*2000);
    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	*(fcfcfc+i)=(double **)malloc(sizeof(double *)*2000);
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    *(*(fcfcfc+i)+j)=(double *)malloc(sizeof(double )*2000);
	}
    }
    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    for(kk=0;kk<*(num+i);kk++){
		k = neighshort[i][kk];
		k &= NEIGHMASK;
		if(k<=j) continue;
		*(*(*(fcfcfc+i)+j)+k)=fc[i][j]*fc[i][k]*fc[j][k];
		*(*(*(fcfcfc+i)+k)+j)=*(*(*(fcfcfc+i)+j)+k);
	    }
	}
    }
    double ***expr2r2r2;
    expr2r2r2=(double ***)malloc(sizeof(double **)*2000);
    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	*(expr2r2r2+i)=(double **)malloc(sizeof(double *)*2000);
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    *(*(expr2r2r2+i)+j)=(double *)malloc(sizeof(double )*2000);
	}
    }
    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    for(kk=0;kk<*(num+i);kk++){
		k = neighshort[i][kk];
		k &= NEIGHMASK;
		if(k<=j) continue;
		*(*(*(expr2r2r2+i)+j)+k)=expr2[i][j]*expr2[i][k]*expr2[j][k];
		*(*(*(expr2r2r2+i)+k)+j)=*(*(*(expr2r2r2+i)+j)+k);
	    }
	}
    }
    double ***F4;
    F4=(double ***)malloc(sizeof(double **)*2000);
    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	*(F4+i)=(double **)malloc(sizeof(double *)*2000);
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    *(*(F4+i)+j)=(double *)malloc(sizeof(double )*2000);
	}
    }
    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    for(kk=0;kk<*(num+i);kk++){
		k = neighshort[i][kk];
		k &= NEIGHMASK;
		if(k<=j) continue;
		*(*(*(F4+i)+j)+k)=pow(*(*(*(expr2r2r2+i)+j)+k),-0.005)* *(*(*(fcfcfc+i)+j)+k);
		*(*(*(F4+i)+k)+j)=*(*(*(F4+i)+j)+k);
	    }
	}
    }

    double costheta;
    double etaa,gammaa,zetaa,minn,maxx,F2[1000],AzF4;
    int itype,jtype,ktype,T2type,T3type,fingeroffset;
    int c;

    // initialize finger prints of all my atoms
    for(i = 0; i < maxlocal; i++) for(j =0; j < MaxNoFingerprints; j++) fingerprint[i][j] = 0.0;
    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	}
    }

    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	itype = map[type[i]];
	// two-body interactions, skip half of them
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    jtype = map[type[j]];
	    // execatly divide pairs into two parts
	    // each pairs will be computed for only once
	    if (j <= i) continue;
	    // compute G2 finger print of i and j with rij
	    c=0;
	    for(nn = 0; nn < G2[itype].NG2; nn++){
		T2type = G2[itype].T2[nn];
		if(T2type != jtype) continue;
		etaa = G2[itype].eta[nn];
		F2[c] = fast_compute_G2(etaa,expr2[i][j],fc[i][j]);
		fingerprint[i][nn]+= F2[c];
		c=c+1;
	    }
	    //fprintf(screen,"HERE! Compute %d %d \n",ii,jj); // for debugging
	    // since each pairs will be computed only once, contribution to finger print of atom j from atom i needs to be computed
	    c=0;
	    for(nn = 0; nn < G2[jtype].NG2; nn++){
		T2type = G2[jtype].T2[nn];
		if(T2type != itype) continue;
		fingerprint[j][nn]+= F2[c];
		c=c+1;
	    }
	}
	// three-body interactions
	// skip immediately if I-J is not within cutoff
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    jtype = map[type[j]];

	    for (kk = 0; kk < *(num+i); kk++) {
		if (jj == kk) continue;
		k = neighshort[i][kk];
		ktype = map[type[k]];
		if(k<=j) continue;

		r=rij[k][j];
		if (r > rc) continue;

		// accumulate G4
		costheta=0.0;
		for(u=0;u<3;u++) costheta=costheta+uij[j][i][u]*uij[k][i][u];

		for(nn = 0; nn < G4[itype].NG4; nn++){
		    T2type = G4[itype].G4T2[nn];
		    T3type = G4[itype].G4T3[nn];
		    if(filter(T2type,T3type,jtype,ktype)<0.0) continue;
		    etaa = G4[itype].G4eta[nn];
		    gammaa = G4[itype].G4gamma[nn];
		    zetaa = G4[itype].G4zeta[nn];
		    AzF4 = fast_fast_compute_G4(etaa,gammaa,zetaa,costheta,*(*(*(F4+i)+j)+k));
		    fingeroffset = G2[itype].NG2;
		    fingerprint[i][fingeroffset+nn]+= AzF4;
		}
	    }
	}
    }
     
    int HLstart,HLend;
    tagint itag;
    double atomE;

    double **NN;
    NN = new double *[MaxNoLayers];
    for (i = 0; i < MaxNoLayers; i++) NN[i] = new double [MaxNoNodes];

    double FinalLayer[2000];
    for (ii = 0; ii < 2000; ii++) FinalLayer[ii]=0.0;

    double *SplitedSubNN;
    SplitedSubNN=(double *)malloc(sizeof(double )*(2000+1)*MaxNoLayers*MaxNoNodes);
    for (ii=0;ii<2000+1;ii++) for (nn = 0; nn < MaxNoLayers; nn++) for(mm = 0; mm < MaxNoNodes; mm++) *(SplitedSubNN+ii*MaxNoLayers*MaxNoNodes+nn*MaxNoNodes+mm)=0.0;

    double *TotalSubNN;
    TotalSubNN=(double *)malloc(sizeof(double )*(2000+1)*MaxNoLayers*MaxNoNodes);
    for (ii=0;ii<2000+1;ii++) for (nn = 0; nn < MaxNoLayers; nn++) for(mm = 0; mm < MaxNoNodes; mm++) *(TotalSubNN+ii*MaxNoLayers*MaxNoNodes+nn*MaxNoNodes+mm)=0.0;

    double SplitedFinalLayer[2000+1];
    for (ii=0;ii<2000+1;ii++) SplitedFinalLayer[ii]=0.0;

    double TotalFinalLayer[2000+1];
    for (ii=0;ii<2000+1;ii++) TotalFinalLayer[ii]=0.0;

    for (ii = 0; ii < inum; ii++){
	i = ilist[ii];
	itag = tag[i];
	itype = map[type[i]];

	HLend = 0;
	for (jj = 0; jj < itype + 1; jj++) HLend+= NoLayers[jj];

	HLstart = HLend - NoLayers[itype]; // find index for starting point of hidden layer
	//fprintf(screen,"type %d, Layer start at %d, end at %d \n",itype,HLstart,HLend);
	// zero out all nodes
	for (nn = 0; nn < MaxNoLayers; nn++) for(mm = 0; mm < MaxNoNodes; mm++) NN[nn][mm] = 0.0;

	for(mm = 0; mm < NoFingerPrints[itype]; mm++){
	    fingeroffset = G2[itype].NG2;
	    if(mm<fingeroffset){
		minn = G2[itype].minima[mm];
		maxx = G2[itype].maxima[mm];
	    }
	    else{
		minn = G4[itype].G4minima[mm-fingeroffset];
		maxx = G4[itype].G4maxima[mm-fingeroffset];
	    }
	    if(maxx - minn > 1.0e-8) fingerprint[i][mm] = -1.0 + 2.0*(fingerprint[i][mm]-minn)/(maxx-minn);
	}
	for(nn = 0; nn < Layer[HLstart + 0].Nnode; nn++){
	    for(mm = 0; mm < NoFingerPrints[itype]; mm++) NN[0][nn]+= Layer[HLstart+0].weight[mm][nn]*fingerprint[i][mm]; //DEBUGGING, force FP=1
	    NN[0][nn] = NN[0][nn] + Layer[HLstart+0].weight[NoFingerPrints[itype]][nn]; // apply bias
	    *(SplitedSubNN+itag*MaxNoLayers*MaxNoNodes+0*MaxNoNodes+nn)=NN[0][nn];
	    NN[0][nn] = tanh(NN[0][nn]); // activation DEBUGGING              
	}
	    
	for (kk = 1; kk < NoLayers[itype]; kk++){
	    for(nn = 0; nn < Layer[HLstart+kk].Nnode; nn++){
		for(mm = 0; mm < Layer[HLstart+kk-1].Nnode; mm++) NN[kk][nn]+= Layer[HLstart+kk].weight[mm][nn]* NN[kk-1][mm];
		NN[kk][nn] = NN[kk][nn] + Layer[HLstart+kk].weight[Layer[HLstart+kk-1].Nnode][nn]; // apply bias
		*(SplitedSubNN+itag*MaxNoLayers*MaxNoNodes+kk*MaxNoNodes+nn)=NN[kk][nn];
		NN[kk][nn] = tanh(NN[kk][nn]); // activation  DEBUGGING
	    }
	}
	// sum over the optput layer
	for(nn = 0; nn < OutLay[itype].NnodeOut; nn++) FinalLayer[ii] = FinalLayer[ii] + OutLay[itype].outweight[nn]* NN[NoLayers[itype]-1][nn];
	FinalLayer[ii] = FinalLayer[ii] + OutLay[itype].outweight[OutLay[itype].NnodeOut]; // apply bias
	SplitedFinalLayer[itag] = FinalLayer[ii]; // apply bias

	// intercept and linear
	atomE = tanh(FinalLayer[ii]); // ?
	atomE = atomE*Linear[itype]+Intercept[itype];
	//fprintf(screen,"atom %d, energy %f \n",ii,atomE);
	if (eflag_global) eng_vdwl += atomE;
	if (eflag_atom) eatom[i] = atomE;
    }
//    printf("E= %.16f\n",eng_vdwl);

    MPI_Allreduce(SplitedSubNN,TotalSubNN,(2000+1)*MaxNoLayers*MaxNoNodes,MPI_DOUBLE,MPI_SUM,world); 
    MPI_Allreduce(SplitedFinalLayer,TotalFinalLayer,2000+1,MPI_DOUBLE,MPI_SUM,world);

    int step;
    int qq,q,qtag,qtype,KK,ktag;
    double dF2dqu[1000],P,r2ij,r2ik;
    double rqj,rqk,rjk;
    double dE;
    double sech2NN;
    double dAF4dqu[1000][2];
    tagint jtag;

    double **dNN; 
    dNN = new double *[MaxNoLayers];
    for (i = 0; i < MaxNoLayers; i++) dNN[i] = new double [MaxNoNodes];

    for(qq=0;qq<inum;qq++){
	q=ilist[qq];
	qtag = tag[q];
	qtype = map[type[q]];
//	printf("%4d       ",qtag);	
	for(u=0;u<3;u++){

	    for(mm=0;mm<G2[qtype].NG2+G4[qtype].NG4;mm++) fingerprintprime[q][mm]=0.0;

	    for(jj=0;jj<*(num+q);jj++){
		j = neighshort[q][jj];
		jtag = tag[j];
		jtype = map[type[j]];

		for(mm=0;mm<G2[jtype].NG2;mm++) fingerprintprime[j][mm]=0.0;

		c=0;
		for(mm = 0; mm < G2[qtype].NG2; mm++){
		    T2type = G2[qtype].T2[mm];
		    if(T2type != jtype) continue;
		    etaa = G2[qtype].eta[mm];
		    dF2dqu[c]=fast_compute_dF2dqu(etaa,fc[q][j],dfc[q][j],expr2[q][j],rij[q][j],uij[q][j][u],rc);
		    fingerprintprime[q][mm]=fingerprintprime[q][mm]+dF2dqu[c];
		    c=c+1;
		}
		c=0;
		for(mm = 0; mm < G2[jtype].NG2; mm++){
		    T2type = G2[jtype].T2[mm];
		    if(T2type != qtype) continue;
		    fingerprintprime[j][mm]=fingerprintprime[j][mm]+dF2dqu[c];
		    c=c+1;
		}
	    }

	    for(jj=0;jj<*(num+q);jj++){
		j = neighshort[q][jj];
		jtag = tag[j];
		jtype = map[type[j]];


		fingeroffset = G2[jtype].NG2;
		for(mm=0;mm<G4[jtype].NG4;mm++) fingerprintprime[j][mm+fingeroffset]=0.0;

		for(KK=0;KK<*(num+q);KK++){
		    k = neighshort[q][KK];
		    ktag = tag[k];
		    ktype = map[type[k]];

		    if (j==k) continue;

		    r=rij[j][k];
		    if (r>rc) continue;

		    c=0;
		    for(mm = 0; mm < G4[qtype].NG4; mm++){
			T2type = G4[qtype].G4T2[mm];
			T3type = G4[qtype].G4T3[mm];
			if(filter(T2type,T3type,jtype,ktype)<0.0) continue;
			etaa = G4[qtype].G4eta[mm];
			gammaa = G4[qtype].G4gamma[mm];
			zetaa = G4[qtype].G4zeta[mm];
			common_fast_fast_compute_dAF4dqu(&dAF4dqu[c][0],q,u,j,k,etaa,gammaa,zetaa,rij,fc,*(*(*(F4+q)+j)+k),dfc,expr2,uij,rc);
			fingeroffset = G2[qtype].NG2;
			fingerprintprime[q][mm+fingeroffset]=fingerprintprime[q][mm+fingeroffset]+dAF4dqu[c][0];
			c=c+1;
		    }

		    c=0;
		    for(mm = 0; mm < G4[jtype].NG4; mm++){
			T2type = G4[jtype].G4T2[mm];
			T3type = G4[jtype].G4T3[mm];
			if(filter(T2type,T3type,qtype,ktype)<0.0) continue;
			fingeroffset = G2[jtype].NG2;
			fingerprintprime[j][mm+fingeroffset]=fingerprintprime[j][mm+fingeroffset]-dAF4dqu[c][1];
			c=c+1;
		    }
		}
		for (nn = 0; nn < MaxNoLayers; nn++) for(mm = 0; mm < MaxNoNodes; mm++) dNN[nn][mm] = 0.0;

		HLend = 0;

		for (ii = 0; ii < jtype + 1; ii++) HLend+= NoLayers[ii];
		HLstart = HLend - NoLayers[jtype]; // find index for starting point of hidden layer

		for(mm = 0; mm < NoFingerPrints[jtype]; mm++){
		    fingeroffset = G2[jtype].NG2;
		    if(mm<fingeroffset){
			minn = G2[jtype].minima[mm];
			maxx = G2[jtype].maxima[mm];
		    }else{
			minn = G4[jtype].G4minima[mm-fingeroffset];
			maxx = G4[jtype].G4maxima[mm-fingeroffset];
		    }
		    if(maxx - minn > 1.0e-8) fingerprintprime[j][mm] = 2.0*fingerprintprime[j][mm]/(maxx-minn);
		}

		for(nn = 0; nn < Layer[HLstart + 0].Nnode; nn++){
		    for(mm = 0; mm < NoFingerPrints[jtype]; mm++){
			dNN[0][nn]+= Layer[HLstart+0].weight[mm][nn]*fingerprintprime[j][mm]; //DEBUGGING, force FP=1
		    }
		    sech2NN = pow(1.0/cosh(*(TotalSubNN+jtag*MaxNoLayers*MaxNoNodes+0*MaxNoNodes+nn)),2.0); // activation DEBUGGING              
		    dNN[0][nn] = dNN[0][nn]*sech2NN;
		}
		for (kk = 1; kk < NoLayers[jtype]; kk++){
		    for(nn = 0; nn < Layer[HLstart+kk].Nnode; nn++){
			for(mm = 0; mm < Layer[HLstart+kk-1].Nnode; mm++) dNN[kk][nn]+= Layer[HLstart+kk].weight[mm][nn]* dNN[kk-1][mm];
			sech2NN = pow(1.0/cosh(*(TotalSubNN+jtag*MaxNoLayers*MaxNoNodes+kk*MaxNoNodes+nn)),2.0); // activation  DEBUGGING
			dNN[kk][nn] = dNN[kk][nn]*sech2NN;
		    }
		}
		dE=0.0;
		for(nn = 0; nn < OutLay[jtype].NnodeOut; nn++) dE = dE + OutLay[jtype].outweight[nn]* dNN[NoLayers[jtype]-1][nn];

		// intercept and linear
		sech2NN = pow(1.0/cosh(TotalFinalLayer[jtag]),2.0); // activation DEBUGGING

		dE = dE*sech2NN*Linear[jtype];
		f[q][u]=f[q][u]-dE;
	    }

	    for (nn = 0; nn < MaxNoLayers; nn++) for(mm = 0; mm < MaxNoNodes; mm++) dNN[nn][mm] = 0.0;

	    HLend = 0;

	    for (jj = 0; jj < qtype + 1; jj++) HLend+= NoLayers[jj];
	    HLstart = HLend - NoLayers[qtype]; // find index for starting point of hidden layer

	    for(mm = 0; mm < NoFingerPrints[qtype]; mm++){
		fingeroffset = G2[qtype].NG2;
		if(mm<fingeroffset){
		    minn = G2[qtype].minima[mm];
		    maxx = G2[qtype].maxima[mm];
		}else{
		    minn = G4[qtype].G4minima[mm-fingeroffset];
		    maxx = G4[qtype].G4maxima[mm-fingeroffset];
		}
		if(maxx - minn > 1.0e-8) fingerprintprime[q][mm] = 2.0*fingerprintprime[q][mm]/(maxx-minn);
	    }
	    for(nn = 0; nn < Layer[HLstart + 0].Nnode; nn++){
		for(mm = 0; mm < NoFingerPrints[qtype]; mm++) dNN[0][nn]+= Layer[HLstart+0].weight[mm][nn]*fingerprintprime[q][mm]; //DEBUGGING, force FP=1
		sech2NN = pow(1.0/cosh(*(TotalSubNN+qtag*MaxNoLayers*MaxNoNodes+0*MaxNoNodes+nn)),2.0); // activation DEBUGGING              
		dNN[0][nn] = dNN[0][nn]*sech2NN;
	    }

	    for (kk = 1; kk < NoLayers[qtype]; kk++){
		for(nn = 0; nn < Layer[HLstart+kk].Nnode; nn++){
		    for(mm = 0; mm < Layer[HLstart+kk-1].Nnode; mm++){
			dNN[kk][nn]+= Layer[HLstart+kk].weight[mm][nn]* dNN[kk-1][mm];
		    }
		    sech2NN = pow(1.0/cosh(*(TotalSubNN+qtag*MaxNoLayers*MaxNoNodes+kk*MaxNoNodes+nn)),2.0); // activation  DEBUGGING
		    dNN[kk][nn] = dNN[kk][nn]*sech2NN;
		}
	    }

	    dE=0.0;
	    for(nn = 0; nn < OutLay[qtype].NnodeOut; nn++) dE = dE + OutLay[qtype].outweight[nn]* dNN[NoLayers[qtype]-1][nn];

	    // intercept and linear
	    sech2NN = pow(1.0/cosh(TotalFinalLayer[qtag]),2.0); // activation DEBUGGING

	    dE = dE*sech2NN*Linear[qtype];
	    f[q][u]=f[q][u]-dE;
	   // printf("     %11.8f",f[q][u]);
	}
//	printf("\n");
    }

    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    free(*(*(fcfcfc+i)+j));
	}
	free(*(fcfcfc+i));
    }
    free(fcfcfc);

    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    free(*(*(expr2r2r2+i)+j));
	}
	free(*(expr2r2r2+i));
    }
    free(expr2r2r2);

    for (ii = 0; ii < inum; ii++) {
	i = ilist[ii];
	for (jj = 0; jj < *(num+i); jj++) {
	    j = neighshort[i][jj];
	    j &= NEIGHMASK;
	    free(*(*(F4+i)+j));
	}
	free(*(F4+i));
    }
    free(F4);

    for(i=0;i<MaxNoLayers;i++) delete [] NN[i];
    delete [] NN;

    for(i=0;i<MaxNoLayers;i++) delete [] dNN[i];
    delete [] dNN;

    free(num);
    free(SplitedSubNN);
    free(TotalSubNN);

}
/* ---------------------------------------------------------------------- */
void PairAnn::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
//  memory->create(neighshort,maxlocal,maxshort,"pair:neighshort");
  map = new int[n+1];

  // **fingerprint? **fingerprintprime?
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairAnn::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");
  cutmax = atof(arg[0]); // cut off distance for ANN
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs 
------------------------------------------------------------------------- */

void PairAnn::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  
  // allocate memory for Linear, Intercept, Activation
  memory->create(Linear,nelements,"pair:Linear");
  memory->create(Intercept,nelements,"pair:Intercept");
  memory->create(NoLayers,nelements,"pair:NoLayers");
  memory->create(NoFingerPrints,nelements,"pair:NoFingerPrints");


  // read potential file and initialize potential parameters

  G2 = (G2FingerPrint *)
    memory->srealloc(G2,nelements*sizeof(G2FingerPrint),"pair:G2FingerPrint");

  G4 = (G4FingerPrint *)
    memory->srealloc(G4,nelements*sizeof(G4FingerPrint),"pair:G4FingerPrint");


  OutLay = (OutLayer *)
    memory->srealloc(OutLay,nelements*sizeof(OutLayer),"pair:OutLayer");

  read_file(arg[2]);

  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairAnn::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Ann requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Ann requires newton pair on");

  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairAnn::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairAnn::read_file(char *file)
{
  int i,j,k,m,n,count,index,nwords;
  int previous,layercount,maxno;
  char *tmp,buf[3],buf2[3];
  char line[MAXLINE];
  int max_para_per_line = 100;
  char **words = new char*[max_para_per_line];

  FILE *fp;

  // HOW TO ALLOCATE MEMORY FOR G2, G4, and Layer? 
  
  
  // open file on proc 0

  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open Ann potential file %s",file);
      error->one(FLERR,str);
    }
  }


  // loop over potential file
  count = 0; 
  layercount = 0; 
  
   if (comm->me ==0) {
        fgets(line,MAXLINE,fp); // read 1st line
        
    }
  
  maxno = 0;

  for (i = 0; i < nelements; i++) {
    G2FingerPrint *G2file = &G2[i]; 
    G4FingerPrint *G4file = &G4[i]; 
    // read the first line
    if (comm->me ==0) {
       // fgets(line,MAXLINE,fp); // read 1st line
        fgets(line,MAXLINE,fp); 
        
    }
    // example

    // 1st line: Ann potential for Sb-MoS2
    // 2nd line: Sb potential
    // The first two lines are ignored by the program.
    // Note that in the pair_coeff * * PotentialFile Sb Mo S
    // The sequence in pair_coeff should follow that in the potential file
    // 
    

    if(comm->me == 0){
        fgets(line,MAXLINE,fp); // read 3rd line:  G2 NG2
        n = strlen(line) + 1;
    }
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);
    sscanf(line,"%s %d",buf,&G2file->NG2);
    
//    fprintf(screen,"NG2 is %d",G2file->NG2);
    memory->create(G2file->T2,(G2file->NG2+1),"pair:T2");
    memory->create(G2file->eta,(G2file->NG2+1),"pair:eta");
    memory->create(G2file->minima,(G2file->NG2+1),"pair:minima");
    memory->create(G2file->maxima,(G2file->NG2+1),"pair:maxima");
    
    // 4th line till end of G2 section:
    // T2 eta minima maxima
    for(j = 0;j < G2file->NG2;j++){
        if(comm->me ==0){
             fgets(line,MAXLINE,fp);
             n = strlen(line) + 1;
        }
        MPI_Bcast(&n,1,MPI_INT,0,world);
        MPI_Bcast(line,n,MPI_CHAR,0,world);
        sscanf(line,"%s %lf %lf %lf",buf,&G2file->eta[j],&G2file->minima[j],&G2file->maxima[j]);
        
        
        for(k = 0; k < nelements; k++){  
          if(strcmp(buf,elements[k]) == 0){
           // fprintf(screen,"%s %s %d \n",buf,elements[k],k);
            G2file->T2[j] = k;
            break;
          }
        }
    }
    
     // Read G4 part of element i

    if(comm->me == 0){
        fgets(line,MAXLINE,fp); // read 1st line of G4: G4 NG4
        n = strlen(line) + 1;
    }
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);
    sscanf(line,"%s %d",buf,&G4file->NG4);

    memory->create(G4file->G4T2,(G4file->NG4+1),"pair:G4T2");
    memory->create(G4file->G4T3,(G4file->NG4+1),"pair:G4T3");
    memory->create(G4file->G4eta,(G4file->NG4+1),"pair:G4eta");
    memory->create(G4file->G4gamma,(G4file->NG4+1),"pair:G4gamma");
    memory->create(G4file->G4zeta,(G4file->NG4+1),"pair:G4zeta");
    memory->create(G4file->G4minima,(G4file->NG4+1),"pair:G4minima");
    memory->create(G4file->G4maxima,(G4file->NG4+1),"pair:G4maxima");
    // fprintf(screen,"HERE! NG4= %d \n",G4file->NG4);
    for(j = 0;j < G4file->NG4;j++){
        if(comm->me ==0){
             fgets(line,MAXLINE,fp);
             n = strlen(line) + 1;
        }
        MPI_Bcast(&n,1,MPI_INT,0,world);
        MPI_Bcast(line,n,MPI_CHAR,0,world);
        sscanf(line,"%s %s %lf %lf %lf %lf %lf",buf,buf2,&G4file->G4eta[j],&G4file->G4gamma[j],&G4file->G4zeta[j],&G4file->G4minima[j],&G4file->G4maxima[j]);
        
        for(k = 0; k < nelements; k++){
          if(strcmp(buf,elements[k]) == 0){
            G4file->G4T2[j] = k;
            break;
          }
        }
        for(k = 0; k < nelements; k++){
          if(strcmp(buf2,elements[k]) == 0){
            G4file->G4T3[j] = k;
            break;
          }
        }
    }
    
    
    int n1,n2;
    n1 = G2file->NG2;
    n2 = G4file->NG4;
    NoFingerPrints[i] = n1 + n2;

    if (maxno <= NoFingerPrints[i]){
      maxno = NoFingerPrints[i];
    }
    // Read neural network weighting matrix! 
    // NLayers 3
    // #Layer 1 weight  
    // Layer 10 
    // ........... 
    // #Layer 2 weight
    // Layer 10 
    // ...........
    // linear intercept XX XX    
    if(comm->me == 0){
        fgets(line,MAXLINE,fp); // read neural network architecture
        n = strlen(line) + 1;
    }
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);
    sscanf(line,"%s %d",buf,&NoLayers[i]);
    
    if (NoLayers[i] > MaxNoLayers) MaxNoLayers = NoLayers[i];

    count = count + NoLayers[i];
    Layer = (HiddenLayer *)
    memory->srealloc(Layer,count*sizeof(HiddenLayer),"pair:HiddenLayer");

    previous = 0;
    for(m=0;m<NoLayers[i];m++){
      index = count - NoLayers[i] + m;
      HiddenLayer *hidden = &Layer[index]; 
      if(comm->me == 0){
        fgets(line,MAXLINE,fp); // # Layer 1 weight, useless
        fgets(line,MAXLINE,fp); // read neural network architecture
        n = strlen(line) + 1;
      }
      MPI_Bcast(&n,1,MPI_INT,0,world);
      MPI_Bcast(line,n,MPI_CHAR,0,world);
      sscanf(line,"%s %d",buf,&hidden->Nnode);

      if (hidden->Nnode > MaxNoNodes) MaxNoNodes = hidden->Nnode;

      if(m == 0){
          hidden->Nnodeminus1 = NoFingerPrints[i];
      }else{
          hidden->Nnodeminus1 = previous;
      } 
      previous = hidden->Nnode; // finish setting up dimension of individual layer
    
      // begin to read weighting matrix of individual layer
      memory->create(hidden->weight,(hidden->Nnodeminus1+2),(hidden->Nnode+2),"pair:weight");
      
      for(j = 0;j < hidden->Nnodeminus1+1;j++){
         if(comm->me ==0){
            fgets(line,MAXLINE,fp); // read neural network architecture
            n = strlen(line) + 1;
         }
         MPI_Bcast(&n,1,MPI_INT,0,world);
         MPI_Bcast(line,n,MPI_CHAR,0,world);
         nwords = 0;
         words[nwords++] = strtok(line," \t\n\r\f");
         while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

         for(k = 0;k < hidden->Nnode;k++){
            hidden->weight[j][k]=atof(words[k]);
         }
      }
      layercount = layercount + 1; 
    }

    // read output layer weight
    OutLayer *Out = &OutLay[i];
    if(comm->me == 0){
      fgets(line,MAXLINE,fp); // read #Output layer
      fgets(line,MAXLINE,fp); // read Output N_outnode
      n = strlen(line) + 1;
    }
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);
    sscanf(line,"%s %d",buf,&Out->NnodeOut);

    memory->create(Out->outweight,(Out->NnodeOut+2),"pair:outweight");
    for(j = 0;j < Out->NnodeOut+1;j++){
        if(comm->me ==0){
             fgets(line,MAXLINE,fp);
             n = strlen(line) + 1;
        }
        MPI_Bcast(&n,1,MPI_INT,0,world);
        MPI_Bcast(line,n,MPI_CHAR,0,world);
        sscanf(line,"%lf",&Out->outweight[j]);
    }
    
    if(comm->me == 0){
      fgets(line,MAXLINE,fp); // read #linear intercept
      fgets(line,MAXLINE,fp); // read linear intercept
      n = strlen(line) + 1;
    }
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);
    sscanf(line,"%lf %lf",&Linear[i],&Intercept[i]);
  }

  
 // OLD READ FILE OF PAIR ANN 
  TotalLayer = layercount;
  MaxNoFingerprints = maxno;
  
/* For double check if everything is right

  for(i=0;i<nelements;i++){
    fprintf(screen,"G2 %s %d: maxima[10]= %lf \n ",elements[i],G2[i].T2[10],G2[i].maxima[10]);
    fprintf(screen,"G4 %s %d %d : maxima[23]= %lf \n ",elements[i],G4[i].G4T2[23],G4[i].G4T3[23],G4[i].G4maxima[23]);
  }

  for(i=0;i<TotalLayer;i++){
    fprintf(screen,"%lf %lf %lf \n",Layer[i].weight[9][0],Layer[i].weight[9][1],Layer[i].weight[9][2]);
  }

  for(i=0;i<nelements;i++){
    fprintf(screen,"%s outweight[9] %lf \n ",elements[i],OutLay[i].outweight[9]);
    
  }
*/
}


double PairAnn::compute_G2(double a, double b, double c)
{
  double etaa,rijsq,rcsq,g2,fc;
  double rij,rc;

  etaa = a;
  rijsq = b;
  rcsq = c;
  
  if(rijsq<=rcsq){
     rij = sqrt(rijsq);
     rc = sqrt(rcsq);
     fc = 0.5*(1+cos(MY_PI*rij/rc));
  }else{
     fc = 0;
  }

  g2 = exp(-1*etaa*(rijsq/rcsq))*fc;


  return g2;
}

double PairAnn::fast_compute_G2(double e, double exp, double fc)
{
  double g2;
  g2 = pow(exp,-e)*fc;

  return g2;
}

double PairAnn::compute_G2_prime(int q,int qtag,int u,int i,int itag,int j,int jtag,double e,double r,double ***R,double rc)
{

    double F2ru;
    double fc,fc_prime,F2_prime;
    double iq,jq;

    fc=(1.0+cos(MY_PI*r/rc))/2.0;
    fc_prime=-(MY_PI/rc*sin(MY_PI*r/rc)/2.0);

    if(itag==qtag) {
	iq=1.0;
	jq=0.0;
    }
    else if(jtag==qtag){
	iq=0.0;
	jq=1.0;
    }
    else{
	iq=0.0;
	jq=0.0;
    }

    F2_prime=-2.0*e*r/rc/rc*exp(-e*r*r/rc/rc)*fc+exp(-e*r*r/rc/rc)*fc_prime; 
    F2ru=F2_prime* *(*(*(R+i)+j)+u)/r*(iq-jq);
    
    return F2ru;
}

double PairAnn::Projector2(int T2,int jt)
{   double v;
    if(jt == T2) {
	v=1.0;
	return v;
    }
    else {
	v=-1.0;
	return v;
    }
}

double PairAnn::compute_G4(double a, double b, double c, double d, double e, double f,
                         double *delrij, double *delrik, double h)
{
  double etaa, gammaa, zetaa, rijsq, riksq, rjksq, cutsq;
  double rij,rik,rjk,rc,fc,costheta,g4;
  
  etaa = a;
  gammaa = b;
  zetaa = c;
  rijsq = d;
  riksq = e;
  rjksq = f;
  cutsq = h;
  
  if(rijsq <= cutsq && riksq<=cutsq && rjksq<=cutsq){
    rij = sqrt(rijsq);
    rik = sqrt(riksq);
    rjk = sqrt(rjksq);
    rc = sqrt(cutsq);
    fc = 0.5*(1+cos(MY_PI*rij/rc));
    fc = fc*0.5*(1+cos(MY_PI*rik/rc));
    fc = fc*0.5*(1+cos(MY_PI*rjk/rc)); 
    costheta = (delrij[0]*delrik[0] + delrij[1]*delrik[1] +
              delrij[2]*delrik[2]) / (rij*rik);
    g4 = pow(2,(1.0-zetaa))*pow((1+gammaa*costheta),zetaa)*exp(-1*etaa*(rijsq+riksq+rjksq)/cutsq)*fc;

  } else{
    g4 = 0;
  }
  
  
  return g4;
}

double PairAnn::fast_compute_G4(double e, double g, double z, double costheta, double expr2r2r2, double fcfcfc)
{
  double g4;
  
    g4 = pow(2.0,(1.0-z))*pow(1.0+g*costheta,z)*pow(expr2r2r2,-e)*fcfcfc;
  
  return g4;
}

double PairAnn::fast_fast_compute_G4(double e, double g, double z, double costheta,double F4)
{
  double g4;
  
    g4 = pow(2.0,(1.0-z))*pow(1.0+g*costheta,z)*F4;
  
  return g4;
}

double PairAnn::compute_G4_prime(int q,int qtag,int u,int i,int itag,int j,int jtag,int k,int ktag,double e,double g,double z,double rij,double rik,double rjk,double ***R,double rc)
{
    int m,mu;
    double iq,jq,kq;
    double F4;
    double fc,fc_prime;
    double F2ij,F2ij_prime;
    double F2ik,F2ik_prime;
    double F2jk,F2jk_prime;
    double costhetaijk,A;
    double uij[3],uik[3],ujk[3];
    double v;

    mu=u;

    fc=(1.0+cos(MY_PI*rij/rc))/2.0;
    fc_prime=-(MY_PI/rc*sin(MY_PI*rij/rc)/2.0);
    F2ij=exp(-e*(rij*rij/rc/rc))*fc;
    F2ij_prime=-2.0*e*rij/rc/rc*exp(-e*rij*rij/rc/rc)*fc+exp(-e*rij*rij/rc/rc)*fc_prime;

    fc=(1.0+cos(MY_PI*rik/rc))/2.0;
    fc_prime=-(MY_PI/rc*sin(MY_PI*rik/rc)/2.0);
    F2ik=exp(-e*(rik*rik/rc/rc))*fc;
    F2ik_prime=-2.0*e*rik/rc/rc*exp(-e*rik*rik/rc/rc)*fc+exp(-e*rik*rik/rc/rc)*fc_prime;

    if(rjk<rc){
	fc=(1.0+cos(MY_PI*rjk/rc))/2.0;
	fc_prime=-(MY_PI/rc*sin(MY_PI*rjk/rc)/2.0);
    }
    else{
	fc=0.0;
	fc_prime=0.0;
    }
    F2jk=exp(-e*(rjk*rjk/rc/rc))*fc;
    F2jk_prime=-2.0*e*rjk/rc/rc*exp(-e*rjk*rjk/rc/rc)*fc+exp(-e*rjk*rjk/rc/rc)*fc_prime;
    
    F4=F2ij*F2ik*F2jk;

    costhetaijk=0.0;
    for(m=0;m<3;m++) costhetaijk=costhetaijk+*(*(*(R+i)+j)+m) * *(*(*(R+i)+k)+m);
    costhetaijk=costhetaijk/rij/rik;
    A=1.0+g*costhetaijk;

    for(m=0;m<3;m++) uij[m]=*(*(*(R+i)+j)+m)/rij;
    for(m=0;m<3;m++) uik[m]=*(*(*(R+i)+k)+m)/rik;
    for(m=0;m<3;m++) ujk[m]=*(*(*(R+j)+k)+m)/rjk;

    if(itag==qtag) {
	iq=1.0;
	jq=0.0;
	kq=0.0;
    }
    else if(jtag==qtag){
	iq=0.0;
	jq=1.0;
	kq=0.0;
    }
    else if(ktag==qtag) {
	iq=0.0;
	jq=0.0;
	kq=1.0;
    }else {
	iq=0.0;
	jq=0.0;
	kq=0.0;
    }
    v=pow(2.0,-z)*F4*z*pow(A,z-1.0)*g*( (uik[mu]-costhetaijk*uij[mu])/rij*(iq-jq)+(uij[mu]-costhetaijk*uik[mu])/rik*(iq-kq) )+pow(2.0,-z)*pow(A,z)*F4*( F2ij_prime/F2ij*uij[mu]*(iq-jq)+F2ik_prime/F2ik*uik[mu]*(iq-kq)+F2jk_prime/F2jk*ujk[mu]*(jq-kq) );
    return v;

}

double PairAnn::filter(int T2,int T3,int jt,int kt)
{
    double v;

    if(jt==T2 && kt==T3) v=1.0;
    else if(jt==T3 && kt==T2) v=1.0;
    else v=-1.0;

    return v;
}

double PairAnn::compute_dF2dqu(int q,int u,int j,double e,double r,double ***R,double rc)
{

    double dF2ru;
    double fc,dfc,dF2;

    fc=(1.0+cos(MY_PI*r/rc))/2.0;
    dfc=-(MY_PI/rc*sin(MY_PI*r/rc)/2.0);
    dF2=-2.0*e*r/rc/rc*exp(-e*r*r/rc/rc)*fc+exp(-e*r*r/rc/rc)*dfc; 
    dF2ru=dF2* *(*(*(R+q)+j)+u)/r;
    
    return dF2ru;
}

double PairAnn::fast_compute_dF2dqu(double e,double fc,double dfc,double expr2,double r,double u,double rc)
{

    double dF2ru;

    dF2ru=pow(expr2,-e)*(dfc-2.0*e*r/rc/rc*fc)*u; 
    
    return dF2ru;
}

double PairAnn::compute_dAF4dqu(int i,int u,int j,int k,double e,double g,double z,double rij,double rik,double rjk,double ***R,double rc)
{
    int m;
    double F4;
    double fc,fc_prime;
    double F2ij,F2ij_prime;
    double F2ik,F2ik_prime;
    double F2jk,F2jk_prime;
    double cosijk,A;
    double uij[3],uik[3],ujk[3];
    double v;

    fc=(1.0+cos(MY_PI*rij/rc))/2.0;
    F2ij=exp(-e*(rij*rij/rc/rc))*fc;
    fc_prime=-(MY_PI/rc*sin(MY_PI*rij/rc)/2.0);
    F2ij_prime=-2.0*e*rij/rc/rc*exp(-e*rij*rij/rc/rc)*fc+exp(-e*rij*rij/rc/rc)*fc_prime;
    
    fc=(1.0+cos(MY_PI*rik/rc))/2.0;
    F2ik=exp(-e*(rik*rik/rc/rc))*fc;

    fc=(1.0+cos(MY_PI*rjk/rc))/2.0;
    F2jk=exp(-e*(rjk*rjk/rc/rc))*fc;
    fc_prime=-(MY_PI/rc*sin(MY_PI*rjk/rc)/2.0);
    F2jk_prime=-2.0*e*rjk/rc/rc*exp(-e*rjk*rjk/rc/rc)*fc+exp(-e*rjk*rjk/rc/rc)*fc_prime;

    F4=F2ij*F2ik*F2jk;

    cosijk=0.0;
    for(m=0;m<3;m++) cosijk=cosijk+*(*(*(R+i)+j)+m) * *(*(*(R+i)+k)+m);
    cosijk=cosijk/rij/rik;
    A=1.0+g*cosijk;

    for(m=0;m<3;m++) uij[m]=*(*(*(R+i)+j)+m)/rij;
    for(m=0;m<3;m++) uik[m]=*(*(*(R+i)+k)+m)/rik;
    for(m=0;m<3;m++) ujk[m]=*(*(*(R+j)+k)+m)/rjk;

    v=pow(2.0,1.0-z)*F4*z*pow(A,z-1.0)*g*(uik[u]-cosijk*uij[u])/rij+pow(2.0,1.0-z)*pow(A,z)*F4*( F2ij_prime/F2ij*uij[u] );

    return v;

}

double PairAnn::fast_compute_dAF4dqu(int i,int u,int j,int k,double e,double g,double z,double r,double fc,double fcfcfc,double dfc,double expr2,double expr2r2r2,double ***uij,double rc)
{
    int m;
    double F4;
    double F2ij,dF2ij;
    double costheta,A;
    double v;

    F2ij=pow(expr2,-e)*fc;
    dF2ij=pow(expr2,-e)*(dfc - 2.0*e*r/rc/rc*fc);
    
    F4=pow(expr2r2r2,-e)*fcfcfc;

    costheta=0.0;
    for(m=0;m<3;m++) costheta=costheta+*(*(*(uij+i)+j)+m) * *(*(*(uij+i)+k)+m);
    A=1.0+g*costheta;

    v=pow(2.0,1.0-z)*F4*pow(A,z)*( z*g/A*(uij[i][k][u]-costheta*uij[i][j][u])/r+dF2ij/F2ij*uij[i][j][u] );

    return v;

}

double PairAnn::fast_fast_compute_dAF4dqu(int i,int u,int j,int k,double e,double g,double z,double r,double fc,double F4,double dfc,double expr2,double ***uij,double rc)
{
    int m;
    double F2ij,dF2ij;
    double costheta,A;
    double v;

    F2ij=pow(expr2,-e)*fc;
    dF2ij=pow(expr2,-e)*(dfc - 2.0*e*r/rc/rc*fc);
    
    costheta=0.0;
    for(m=0;m<3;m++) costheta=costheta+*(*(*(uij+i)+j)+m) * *(*(*(uij+i)+k)+m);
    A=1.0+g*costheta;

    v=pow(2.0,1.0-z)*F4*pow(A,z)*( z*g/A*(uij[i][k][u]-costheta*uij[i][j][u])/r+dF2ij/F2ij*uij[i][j][u] );

    return v;

}

//double PairAnn::common_fast_fast_compute_dAF4dqu123(double *dAF4dqu,int i,int u,int j,int k,double e,double g,double z,double **r,double **fc,double F4,double **dfc,double **expr2,double ***uij,double rc)
//{   
//    int m;   
//    double F2ij,dF2ij;
//    double F2jk,dF2jk;
//    double costhetaqjk,Aqjk,Aqjkz;
//    double costhetajqk,Ajqk,Ajqkz;
//    double constant,common1,common2,v;
//
//    constant=pow(2.0,1.0-z)*F4;
//
//    F2ij=pow(expr2[i][j],-e)*fc[i][j];
//    dF2ij=pow(expr2[i][j],-e)*( dfc[i][j]-2.0*e* r[i][j]/rc/rc*fc[i][j] );
//    common1=dF2ij/F2ij*uij[i][j][u];
//
//    common2=uij[i][j][u]/rij[i][j];
//
//    costhetaqjk=0.0; 
//    for(m=0;m<3;m++) costhetaqjk=costhetaqjk+uij[i][j][m]*uij[i][k][m];
//    Aqjk=1.0+g*costhetaqjk;
//    Aqjkz=pow(Aqjk,z);
//
//    *(dAF4dqu+0)=constant*Aqjkz*(g*z/Aqjk*( uij[i][k][u]/rij[i][j]-costhetaqjk*common2 )+common1);
//
//    F2jk=pow(*(*(expr2+i)+k),-e)*fc[i][k];
//    dF2jk=pow(*(*(expr2+i)+k),-e)*(dfc[i][k]-2.0*e*r[i][k]/rc/rc*fc[i][k]);
//
//    costhetajqk=0.0; 
//    for(m=0;m<3;m++) costhetajqk=costhetajqk+uij[j][i][m]*uij[j][k][m];
//    Ajqk=1.0+g*costhetajqk;
//    Ajqkz=pow(Ajqk,z);
//
//    *(dAF4dqu+1)=constant*Ajqkz*(g*z/Ajqk*( uij[j][k][u]/rij[i][j]+costhetajqk*common2 )-common1-dF2jk/F2jk*uij[i][k][u]);
//
//    return v;
//}

double PairAnn::j_compute_dAF4dqu(int i,int u,int j,int k,double e,double g,double z,double rij,double rik,double rjk,double ***R,double rc)
{
    int m;
    double F4;
    double fc,fc_prime;
    double F2ij,F2ij_prime;
    double F2ik,F2ik_prime;
    double F2jk,F2jk_prime;
    double cosijk,A;
    double uij[3],uik[3],ujk[3];
    double v;

    fc=(1.0+cos(MY_PI*rij/rc))/2.0;
    F2ij=exp(-e*(rij*rij/rc/rc))*fc;
    fc_prime=-(MY_PI/rc*sin(MY_PI*rij/rc)/2.0);
    F2ij_prime=-2.0*e*rij/rc/rc*exp(-e*rij*rij/rc/rc)*fc+exp(-e*rij*rij/rc/rc)*fc_prime;
    
    fc=(1.0+cos(MY_PI*rik/rc))/2.0;
    F2ik=exp(-e*(rik*rik/rc/rc))*fc;

    fc=(1.0+cos(MY_PI*rjk/rc))/2.0;
    F2jk=exp(-e*(rjk*rjk/rc/rc))*fc;
    fc_prime=-(MY_PI/rc*sin(MY_PI*rjk/rc)/2.0);
    F2jk_prime=-2.0*e*rjk/rc/rc*exp(-e*rjk*rjk/rc/rc)*fc+exp(-e*rjk*rjk/rc/rc)*fc_prime;

    F4=F2ij*F2ik*F2jk;

    cosijk=0.0;
    for(m=0;m<3;m++) cosijk=cosijk+*(*(*(R+i)+j)+m) * *(*(*(R+i)+k)+m);
    cosijk=cosijk/rij/rik;
    A=1.0+g*cosijk;

    for(m=0;m<3;m++) uij[m]=*(*(*(R+i)+j)+m)/rij;
    for(m=0;m<3;m++) uik[m]=*(*(*(R+i)+k)+m)/rik;
    for(m=0;m<3;m++) ujk[m]=*(*(*(R+j)+k)+m)/rjk;

    v=pow(2.0,1.0-z)*F4*z*pow(A,z-1.0)*g*(uik[u]-cosijk*uij[u])/rij+pow(2.0,1.0-z)*pow(A,z)*F4*(F2ij_prime/F2ij*uij[u]-F2jk_prime/F2jk*ujk[u]);

    return v;

}

double PairAnn::fast_j_compute_dAF4dqu(int i,int u,int j,int k,double e,double g,double z,double **r,double **fc,double fcfcfc,double **dfc,double **expr2,double expr2r2r2,double ***uij,double rc)
{
    int m;
    double F4;
    double F2ij,dF2ij;
    double F2jk,dF2jk;
    double costheta,A;
    double v;

    F2ij=pow(*(*(expr2+i)+j),-e) * *(*(fc+i)+j);
    dF2ij=pow(*(*(expr2+i)+j),-e) * ( *(*(dfc+i)+j)-2.0*e* *(*(r+i)+j)/rc/rc* *(*(fc+i)+j) );
    
    F2jk=pow(*(*(expr2+j)+k),-e) * *(*(fc+j)+k);
    dF2jk=pow(*(*(expr2+j)+k),-e) * ( *(*(dfc+j)+k)-2.0*e* *(*(r+j)+k)/rc/rc* *(*(fc+j)+k) );

    F4=pow(expr2r2r2,-e)*fcfcfc;

    costheta=0.0;
    for(m=0;m<3;m++) costheta=costheta+*(*(*(uij+i)+j)+m) * *(*(*(uij+i)+k)+m);
    A=1.0+g*costheta;

    v=pow(2.0,1.0-z)*F4*pow(A,z)*( g*z/A*( *(*(*(uij+i)+k)+u)-costheta**(*(*(uij+i)+j)+u))/rij[i][j]+dF2ij/F2ij**(*(*(uij+i)+j)+u)-dF2jk/F2jk**(*(*(uij+j)+k)+u) );

    return v;

}

double PairAnn::fast_fast_j_compute_dAF4dqu(int i,int u,int j,int k,double e,double g,double z,double **r,double **fc,double F4,double **dfc,double **expr2,double ***uij,double rc)
{
    int m;
    double F2ij,dF2ij;
    double F2jk,dF2jk;
    double costheta,A;
    double v;

    F2ij=pow(*(*(expr2+i)+j),-e) * *(*(fc+i)+j);
    dF2ij=pow(*(*(expr2+i)+j),-e) * ( *(*(dfc+i)+j)-2.0*e* *(*(r+i)+j)/rc/rc* *(*(fc+i)+j) );
    
    F2jk=pow(*(*(expr2+j)+k),-e) * *(*(fc+j)+k);
    dF2jk=pow(*(*(expr2+j)+k),-e) * ( *(*(dfc+j)+k)-2.0*e* *(*(r+j)+k)/rc/rc* *(*(fc+j)+k) );

    costheta=0.0;
    for(m=0;m<3;m++) costheta=costheta+*(*(*(uij+i)+j)+m) * *(*(*(uij+i)+k)+m);
    A=1.0+g*costheta;

    v=pow(2.0,1.0-z)*F4*pow(A,z)*( g*z/A*( *(*(*(uij+i)+k)+u)-costheta**(*(*(uij+i)+j)+u))/rij[i][j]+dF2ij/F2ij**(*(*(uij+i)+j)+u)-dF2jk/F2jk**(*(*(uij+j)+k)+u) );

    return v;
}
void PairAnn::common_fast_fast_compute_dAF4dqu(double *dAF4dqu,int i,int u,int j,int k,double e,double g,double z,double **r,double **fc,double F4,double **dfc,double **expr2,double ***uij,double rc)
{
    int m;
    double F2ik,dF2ik;
    double F2ij,dF2ij;
    double F2jk;
    double costhetaqjk,Aqjk,Aqjkz,Aqjkz_minus_1;
    double costhetajqk,Ajqk,Ajqkz,Ajqkz_minus_1;

    double constant,common1,common2,v;

    constant=pow(2.0,1.0-z)*F4;

    F2ik=pow(expr2[i][k],-e)*fc[i][k];
    dF2ik=pow(expr2[i][k],-e)*( dfc[i][k]-2.0*e* r[i][k]/rc/rc*fc[i][k] );

    F2ij=pow(expr2[i][j],-e)*fc[i][j];
    dF2ij=pow(expr2[i][j],-e)*( dfc[i][j]-2.0*e* r[i][j]/rc/rc*fc[i][j] );

    F2jk=pow(expr2[j][k],-e)*fc[j][k];

    common1=dF2ij/F2ij*uij[i][j][u];

    common2=uij[i][j][u]/rij[i][j];

    costhetaqjk=0.0;
    for(m=0;m<3;m++) costhetaqjk=costhetaqjk+uij[i][j][m]*uij[i][k][m];
    Aqjk=1.0+g*costhetaqjk;
    Aqjkz=pow(Aqjk,z);
    //    *(dAF4dqu+0)=constant*Aqjkz*(g*z/Aqjk*( uij[i][k][u]/rij[i][j]-costhetaqjk*common2 )+common1);

        Aqjkz_minus_1=pow(Aqjk,z-1.0);
        //    *(dAF4dqu+0)=constant*Aqjkz_minus_1*(g*z*( uij[i][k][u]/rij[i][j]-costhetaqjk*common2 )+Aqjk*common1);
            *(dAF4dqu+0)=constant*Aqjkz_minus_1*g*z*( uij[i][k][u]/rij[i][j]-costhetaqjk*common2 )+pow(2.0,1.0-z)*Aqjkz_minus_1*Aqjk*F2ik*F2jk*dF2ij*uij[i][j][u];


            //    F2jk=pow(*(*(expr2+i)+k),-e)*fc[i][k];
            //    dF2jk=pow(*(*(expr2+i)+k),-e)*(dfc[i][k]-2.0*e*r[i][k]/rc/rc*fc[i][k]);

                costhetajqk=0.0;
                    for(m=0;m<3;m++) costhetajqk=costhetajqk+uij[j][i][m]*uij[j][k][m];
                        Ajqk=1.0+g*costhetajqk;
                        //    Ajqkz=pow(Ajqk,z);
                        //    *(dAF4dqu+1)=constant*Ajqkz*(g*z/Ajqk*( uij[j][k][u]/rij[i][j]+costhetajqk*common2 )-common1-dF2jk/F2jk*uij[i][k][u]);

                            Ajqkz_minus_1=pow(Ajqk,z-1.0);
                            //    *(dAF4dqu+1)=constant*Ajqkz_minus_1*(g*z*( uij[j][k][u]/rij[i][j]+costhetajqk*common2 )-Ajqk*(common1+dF2jk/F2jk*uij[i][k][u]) );
                                *(dAF4dqu+1)=constant*Ajqkz_minus_1*g*z*( uij[j][k][u]/rij[i][j]+costhetajqk*common2 )-pow(2.0,1.0-z)*Ajqkz_minus_1*Ajqk*(F2jk*F2ik*dF2ij*uij[i][j][u]+F2ij*F2jk*dF2ik*uij[i][k][u]);

                                }
