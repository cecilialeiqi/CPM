#include<iostream>
#include<fstream>
#include"omp.h"
#include<vector>
#include<cmath>
#include<time.h>
#include<eigen/Eigen/Dense>
#include<eigen/Eigen/Sparse>
#include<eigen/Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

typedef vector<pair <int,double> > Sample;
class smat{
    public:
    vector<vector<pair <int,double> > > mat;
	vector<double> diagonal;
    int n;
    int nnz;
    void read(char* filename);
	void read2(char* filename);
    double* multiple(double* x);
};

void smat::read(char* filename){
    nnz=0;
	ifstream f(filename);
	f>>n;
	for (int i=0;i<n;i++)
		diagonal.push_back(0);
    int i,j;
    double v;
    clock_t st = clock();
    for (i=0;i<n;i++){
		mat.push_back(vector<pair <int,double> >());
	}
    while (!f.eof()){
        f>>i;
        if (f.eof()) break;
        f>>j;
        if (n<j)
            n=j;
        f>>v;
		vector<int>::size_type ix=i-1;
		if (i==j)
		{
			diagonal[ix]=v;
		}
        mat[ix].push_back(make_pair(j-1,v));
        nnz++;
    }

    clock_t end = clock();
    cout<<"reading file time: "<<double(end-st)/CLOCKS_PER_SEC<<endl;
    cout<<" n="<<n<<" nnz="<<nnz<<endl;
};

void smat::read2(char* filename){
    nnz=0;
	ifstream f(filename);
	f>>n;
	for (int i=0;i<n;i++)
		diagonal.push_back(0);
    int i,j;
    double v;
    clock_t st = clock();
    for (i=0;i<n;i++){
		mat.push_back(vector<pair <int,double> >());
	}
	for (int i=0;i<n;i++)
		for (int j=0;j<n;j++)
		{
			f>>v;
			if (abs(v)>0.0000001){//non zero
				nnz++;
				vector<int>::size_type ix=i;
				if (i==j)
				{
					diagonal[ix]=v;
				}
		        mat[ix].push_back(make_pair(j,v));
			}
		}
    clock_t end = clock();
    cout<<"reading file time: "<<double(end-st)/CLOCKS_PER_SEC<<endl;
    cout<<" n="<<n<<" nnz="<<nnz<<endl;
};
void swap(double * a, double * b) {
  double tmp = * a;
  * a = * b;
  * b = tmp;
}

void swap(int * a, int * b) {
  int tmp = * a;
  * a = * b;
  * b = tmp;
}

double* smat::multiple(double* x){
    double *y =new double[n];
//#pragma omp parallel
//    {

//#pragma omp for schedule(dynamic,1000) nowait
    for (int i=0;i<n;i++){
        y[i] = 0;
        Sample* row = &mat[i];
        for (Sample::iterator ii = row->begin();ii != row->end(); ++ii){
            y[i] += ii->second * x[ii->first];
        }
    }
//    }
    return y;
}
