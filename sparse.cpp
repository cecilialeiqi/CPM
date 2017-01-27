#include <algorithm>    // std::nth_element
#include <cstdlib>
#include <stdio.h>
#include "sparse.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <iomanip>
#include <vector>

#pragma GCC diagnostic ignored "-Wwrite-strings"
using namespace std;
typedef Eigen::SparseMatrix<double,ColMajor> SpMat;
typedef Eigen::Triplet<double> T;

ofstream ofile("./result/output.csv");

void read(vector<T>& A_in_triplets, char* filename)
{
	ifstream f(filename);
	long num=0;
	long maxn=0;
	long i,j,v;
    clock_t st = clock();
    while (!f.eof()){
        f>>i;
        if (f.eof()) break;
        f>>j;
        if (f.eof()) break;
		num++;
		A_in_triplets.push_back(T(i-1,j-1,1));
		A_in_triplets.push_back(T(j-1,i-1,1));
		if (num-long(num/1000000)*1000000==0)
		{
			cout<<"Reading "<<num<<": "<<i<<' '<<j<<" 1"<<endl;
		}
		if (i>maxn)
			maxn=i;
		if (j>maxn)
			maxn=j;
    }
	cout<<"maxn="<<maxn<<endl;
    clock_t end = clock();
    cout<<"reading file time: "<<double(end-st)/CLOCKS_PER_SEC<<endl;
}

double func(SpMat & A, VectorXd & x, VectorXd &y)
{
    double f=pow(A.norm(),2);
	f-=2*x.dot(A*y);
	f+=x.dot(x)*y.dot(y);
	return f;
}

void eigens(SpMat & A, VectorXd & x)
{//get ground truth first

	double time=0;
	clock_t st=clock();
	double lastf=0;
	int iter=0;
	for (iter=0; iter<100; iter++)
	{
	     x=A*x;
	     x=x/x.norm();
	}
	clock_t end=clock();
    cout<<"To get ground truth: "<<double(end-st)/CLOCKS_PER_SEC<<" seconds"<<endl;
}


void eigens_pm(SpMat & A, VectorXd x, VectorXd &zn, double & time)
{
	time=0;
    double st = omp_get_wtime();
	double lastf=0;
	int iter=0;
	double c=x.dot(zn);
	ofile<<"Power method,0,"<<sqrt(1-c*c)/c<<endl;
	for (iter=0; iter<1000; iter++)
	{
		x=A*x;
		x=x/x.norm();
		double end = omp_get_wtime();
		time+=double(end-st);
	    c=x.dot(zn);//cosine
		ofile <<","<< std::fixed << std::setw( 11 ) << std::setprecision(6)
          <<time<<','<<sqrt(1-c*c)/c<<endl;
	    if (abs(x.dot(zn)/x.norm())>0.9999999)
			break;
	    st=omp_get_wtime();
	}
    cout<<"Power method time: "<<time<<" seconds"<<endl;
}


double get_cosine(vector<double> & alphas, vector<double> & betas, vector<VectorXd> & V, VectorXd & zn, int m)
{
	MatrixXd T=MatrixXd::Constant(m,m,0);
    T(0,0)=alphas[0];
	for (int i=1;i<m;i++){
	    T(i,i)=alphas[i];
		T(i-1,i)=betas[i-1];
		T(i,i-1)=betas[i-1];
	}
	EigenSolver<MatrixXd> es(T);
	int index=0;
	double largeste=0;
	int j=0;
	for (j=0;j<m;j++)
		if (es.eigenvalues()[j].real()>largeste){
			largeste=es.eigenvalues()[j].real();
			index=j;
		}

	VectorXcd a=es.eigenvectors().col(index);
    VectorXd get_v=a(0).real()*V[0];
    for (int i=1;i<V.size();i++)
       get_v+=a(i).real()*V[i];
    return get_v.dot(zn)/get_v.norm();
}

void eigens_lanczos(SpMat & A, VectorXd v0, VectorXd & zn, double & time)
{
	int n=A.cols();
	time=0;
    double st=omp_get_wtime();
    vector<double> alphas;
    vector<double> betas;
    vector<VectorXd> V;
    VectorXd v1=v0/v0.norm();
    VectorXd f = A*v1;
    double c=v0.dot(zn);
	ofile<<"Lanczos method,0,"<<sqrt(1-c*c)/c<<endl;
	double alpha = v1.dot(f);
    f = f - v1*alpha;
    V.push_back(v1);
 	alphas.push_back(alpha);
	double lastc=0;
    for (int i=1;i<n;i++){
        betas.push_back(f.norm());
        v0 = v1;
        v1 = f/betas[betas.size()-1];
        f = A*v1 - v0*betas[betas.size()-1];
        alphas.push_back(v1.dot(f));
        f = f - v1*alphas[alphas.size()-1];
        V.push_back(v1);
        double end=omp_get_wtime();
		c=get_cosine(alphas,betas,V,zn,i+1);//postprocessing step after early termination
        ofile <<","<< std::fixed << std::setw( 11 ) << std::setprecision(6)<< time+omp_get_wtime()-st<<','<<sqrt(1-c*c)/c<<endl;//To show the relation between current accuracy and time spent, we output the time inside the for loop plus the time of postprocessing after early termination.
		time+=end-st;//We make sure for each iteration, the preprocessing time is only added once. So the "time" variable only saves the time spent inside the for loop
        if (abs(c-lastc)<1e-9 || abs(c)>0.999999)
            break;
		lastc=abs(c);
        st=omp_get_wtime();
	}
	st=omp_get_wtime();
    get_cosine(alphas,betas,V,zn,alphas.size());
    time+=omp_get_wtime()-st;
	cout<<"Lanczos method time: "<<time<<" seconds"<<endl;
}


void eigens_vrpca(SpMat & X, VectorXd v0, VectorXd & zn, double & time)
{
    int n=X.cols();
    time=0;
    double st=omp_get_wtime();
    int m=n;
    VectorXd tmp(n);
	double c=v0.dot(zn);
    ofile<<"VR-PCA"<<",0,"<< sqrt(1-c*c)/c<<endl;
	for (int i=0;i<n;++i){
       tmp[i]=0;
		for (SparseMatrix<double, ColMajor>::InnerIterator it(X,i); it; ++it)
			tmp[i]+=it.value()*it.value();
    }
    double eta=1/(tmp.mean()*sqrt(n));
    VectorXd w=v0/v0.norm();
    for (int i=0;i<100;i++){
        VectorXd u=X*(X.transpose()*w)/n;
        /*for (int j=0;j<m;j++){
	        int s=rand()%n;
			double a=0;
			for (SparseMatrix<double, ColMajor>::InnerIterator it(X,s); it; ++it)
				a+=it.value()*(wrun[it.row()]-w[it.row()]);
			double b=eta*a;
			for (SparseMatrix<double, ColMajor>::InnerIterator it(X,s); it; ++it)
				wrun[it.row()]+=b*it.value();
			wrun+=eta*u;
            //wrun+=eta*(X.col(s).dot(wrun-w)*X.col(s)+u);//dense update
            wrun/=wrun.norm();
        }//dense update*/
		double alpha=1;
		double beta=0;
		VectorXd g=w;
		double gamma=g.dot(g);
		double delta=g.dot(u);
		double zeta=u.dot(u);
		for (int j=0;j<m;j++){
			int s=rand()%n;
			double a=0;
			for (SparseMatrix<double, ColMajor>::InnerIterator it(X,s); it; ++it)
				a+=it.value()*(alpha*g[it.row()]+beta*u[it.row()]-w[it.row()]);
			SparseVector<double> dg=X.col(s)*eta*a;
			for (SparseVector<double>::InnerIterator it(dg); it; ++it)
			{
				g[it.index()]+=it.value()/alpha;
			}
			//g+=dg/alpha;
			beta+=eta;
			gamma+=2*alpha*dg.dot(g)+pow(dg.norm(),2);
			delta+=dg.dot(u);
			double tmp=sqrt(gamma+2*delta*beta+beta*beta*zeta);
			alpha/=tmp;
			beta/=tmp;
			gamma/=tmp*tmp;
			delta/=tmp;
		}
		w=alpha*g+beta*u;
        time+=omp_get_wtime()-st;
		c=w.dot(zn);
	    ofile<< ","<< std::fixed << std::setw( 11 ) << std::setprecision(6)
          <<time<<','<<sqrt(1-c*c)/c<<endl;//CLOCKS_PER_SEC

        if (abs(c)>0.999999){
             break;
		if (time>1200)//stop after 2 hours
			break;
        }
        st=omp_get_wtime();
    }
	cout<<"VR-PCA time: "<<time<<" seconds"<<endl;
}


void eigens_cpm(SpMat & A, VectorXd x, long k, VectorXd & zn, double time)
{
	long n=A.cols();
	time=0;
	double st = omp_get_wtime();
	VectorXd Ax=A*x;
	VectorXd nextx=Ax;
	nextx/=Ax.norm();
	VectorXd wx=x-nextx;
	double lastf=0;
	double c=x.dot(zn);
	ofile<<"cpm,0,"<<sqrt(1-c*c)/c<<endl;
    double * choice=new double[n];
	int iter;
	for (iter=0; iter<1000*2; iter++)
	{
		////////update X

		//choose important k coordinates
	    for (int s=0;s<n;s++)
	    {
			choice[s]=abs(wx[s]);
	    }
	    std::nth_element (choice, choice+n-1-k, choice+n-1);
	    double pivot=choice[n-1-k];
	    for (long i=0;i<n;i++)
	    {
		   if (abs(wx[i])>=pivot)
		   {
			   double dx=nextx[i]-x[i];
			   x[i]=nextx[i];
			   for (SparseMatrix<double,ColMajor>::InnerIterator it(A,i); it; ++it)
			   {
				   Ax[it.row()]+=dx*it.value();
			   }
		   }
	   }

	   nextx=Ax/Ax.norm();
	   wx=x-nextx;
	   double end=omp_get_wtime();
	   time+=double(end-st);
	   c=Ax.dot(zn)/Ax.norm();
		ofile <<','<< std::fixed << std::setw( 11 ) << std::setprecision(6)<<time<<','<<sqrt(1-c*c)/c<<endl; //CLOCKS_PER_SEC
	    if (abs(Ax.dot(zn)/Ax.norm())>0.999999)
			break;
		st = omp_get_wtime();
	}
    cout<<"Coordinate-wise power method time: "<<time<<" seconds"<<endl;
}

void getnewx(SpMat &A, double & xTx, VectorXd &x, VectorXd &Ax, long t)//t-th coordinate
{
			double p=xTx-x[t]*x[t];
			double q=-Ax[t];  //q=-(a_t.*x-A(t,t)*x(t))
			double delta=pow(q/2,2)+pow(p/3,3);
			if (delta>0)
			{
				double a=-q/2+sqrt(delta);
				double term1=pow(abs(a),1.0/3)*a/abs(a);
				a=-q/2-sqrt(delta);
				double term2=pow(abs(a),1.0/3)*a/abs(a);
				xTx-=pow(x[t],2);
				x[t]=term1+term2;
				xTx+=pow(x[t],2);
			}else
			{
				double theta=acos(-q*sqrt(-27*p)/2/p/p);
				double xx[3];
				xx[0]=2*sqrt(-p/3)*cos(theta/3);
				xx[1]=2*sqrt(-p/3)*cos(theta/3+3.14159265359*2/3);
                xx[2]=2*sqrt(-p/3)*cos(theta/3-3.14159265359*2/3);
				double dfbound=0;
				double oldx=x[t];
				for (int s=1;s<3;s++)
				{
                    double df=pow(xx[s]-oldx,2)*(-pow(xx[s]+oldx,2)-2*(xTx-pow(oldx,2)+pow(xx[s],2)));
					if (df<dfbound)
					{
						x[t]=xx[s];
						dfbound=df;
					}
				}
				xTx+=(x[t]+oldx)*(x[t]-oldx);
			}
}


void eigens_sgcd(SpMat & A, VectorXd x, long k, VectorXd &zn, double time)
{
	long n=A.cols();
	time=0;
	double c=x.dot(zn);
	x=A*x;
	x/=x.norm();
	double st = omp_get_wtime();
	VectorXd Ax=A*x;
	double xTx=1;
	ofile<<"Symmetric Greedy Coordinate Descent,0,"<<sqrt(1-c*c)/c<<endl;
	VectorXd wx=x-Ax/xTx;
	double lastf=0;
	int iter;
	double * choice=new double[n];
	for (iter=0; iter<1000*2; iter++)
	{
		////////update X
	    for (int s=0;s<n;s++)
	    {
		choice[s]=abs(wx[s]);
	    }
	     std::nth_element (choice, choice+n-1-k, choice+n-1);
	    double pivot=choice[n-1-k];

	   for (long i=0;i<n;i++)
	   {
		   if (abs(wx[i])>=pivot)
		   {
			   double oldx=x[i];
			   getnewx(A, xTx, x, Ax, i); //xTx is already changed in this function
			   double dx=x[i]-oldx;
			   for (SparseMatrix<double,ColMajor>::InnerIterator it(A,i); it; ++it)
			   {
				   Ax[it.row()]+=dx*it.value();
			   }
		   }
	   }
	   wx=x-Ax/xTx;
	   double end = omp_get_wtime();
	   time+=double(end-st);
	   c=Ax.dot(zn)/Ax.norm();
	   ofile << ","<< std::fixed << std::setw( 11 ) << std::setprecision(6) <<time<<','<<sqrt(1-c*c)/c<<endl;//CLOCKS_PER_SEC
	    if (abs(Ax.dot(zn)/Ax.norm())>0.999999)
			break;
	    st = omp_get_wtime();
	}
    cout<<"Symmetric greedy coordinate descent time: "<<time<< " seconds"<<endl;
}

int main(int argc, char* argv[]){
	if (argc!=3){
        cout<<"usage: "<<argv[0]<<" <filename> <n>"<<endl;
        exit(0);
    }

	char* filename = argv[1];
	long n=atoi(argv[2]);
	ifstream f(filename);
	SpMat A(n,n);
	cout<<A.cols()<<endl;
	vector<T> A_in_triplets;
         read(A_in_triplets,filename);
	A.setFromTriplets(A_in_triplets.begin(), A_in_triplets.end());
	cout<<"Finish establishing the matrix"<<endl;
	VectorXd x(n);
	VectorXd y(n);
	VectorXd zn(n);
	x.setRandom(x.rows(),1);
    x=x/x.norm();
    zn=x;
	eigens(A,zn);
	cout<<"Begin running!"<<endl;
	double time_pm,time_cpm,time_sgcd,time_lanczos,time_vrpca;
	eigens_pm(A, x ,zn,time_pm);
	eigens_cpm(A, x, n/20,zn,time_cpm);
	eigens_sgcd(A,x,n/20,zn,time_sgcd);
	eigens_lanczos(A,x,zn,time_lanczos);
	eigens_vrpca(A,x,zn,time_vrpca);
	ofile.close();
	/////////////////////////////////////
}
