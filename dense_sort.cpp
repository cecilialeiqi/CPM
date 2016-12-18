#include <algorithm>    // std::sort
#include  <cstdlib>
#include <stdio.h>
#include "dense.h"
#include<iostream>
#include<cmath>
#include<omp.h>
#include <iomanip>
#include<cstring>
#include <vector>


#pragma GCC diagnostic ignored "-Wwrite-strings"
using namespace std;

double eigens_pm(MatrixXd & A, VectorXd x, VectorXd &zn,int & flops)
{
	double time=0;
	double st=omp_get_wtime();
	int iter=0;
	for (iter=0; iter<1000; iter++)
	{
		x=A*x;
		x=x/x.norm();
		time+=double(omp_get_wtime()-st);

	    double f=x.dot(zn);
	    if (abs(f)>=0.999999)
			break;
	    st=omp_get_wtime();
	}
	flops=(iter+1)*(A.cols()+1);
	return time;
}

double eigens_gpm(MatrixXd & A, VectorXd x, int k, VectorXd &zn, int & flops)
{
	int n=A.cols();
	flops=n;
	double time=0;
	double st=omp_get_wtime();
	double xTx=pow(x.norm(),2);
	VectorXd Ax=A*x;
	VectorXd nextx=Ax/sqrt(xTx);
	double * wx=new double[n];
	for (int i=0;i<n;i++)
		wx[i]=abs(x[i]-nextx[i]);
	double * tmp=new double[n];
	int iter=0;
	for (iter=0; iter<1000*n/k; iter++)
	{
		////////update X
		memcpy(tmp,wx,sizeof(double)*n);
		std::nth_element (tmp, tmp+n-1-k, tmp+n-1);
	   double pivot=tmp[n-1-k];
	   for (int i=0;i<n;i++)
	   {
		  if (wx[i]>=pivot)
		   {
				//flops++;
			   xTx-=x[i]*x[i];
			   Ax+=A.col(i)*(nextx[i]-x[i]);
			   x[i]=nextx[i];
			   xTx+=x[i]*x[i];
		   }
	   }
	   nextx=Ax/sqrt(xTx);
	   for (int i=0;i<n;i++)
	   wx[i]=abs(x[i]-nextx[i]);
	    time+=double(omp_get_wtime()-st);
	    flops+=2+k;
		double f=Ax.dot(zn)/Ax.norm();
	    if (abs(f)>0.999999)
			break;
	   st=omp_get_wtime();
	}
	return time;
}



void getnewx(MatrixXd &A, double & xTx, VectorXd &x, VectorXd &Ax, int t)//t-th coordinate
{
			double p=xTx-pow(x[t],2)-A(t,t);
			double q=A(t,t)*x[t]-Ax[t];  //q=-(a_t.*x-A(t,t)*x(t))
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
				double theta=acos(-q*sqrt(-27*p)/2/pow(p,2));
				double xx[3];
				xx[0]=2*sqrt(-p/3)*cos(theta/3);
				xx[1]=2*sqrt(-p/3)*cos(theta/3+3.14159265359*2/3);
                xx[2]=2*sqrt(-p/3)*cos(theta/3-3.14159265359*2/3);
				double dfbound=0;
				double oldx=x[t];
				for (int s=1;s<3;s++)
				{
                    double df=pow(xx[s]-oldx,2)*(2*A(t,t)-pow(xx[s]+oldx,2)-2*(xTx-pow(oldx,2)+pow(xx[s],2)));
					if (df<dfbound)
					{
						x[t]=xx[s];
						dfbound=df;
					}
				}
				xTx+=-pow(oldx,2)+pow(x[t],2);
			}
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
	VectorXcd a=es.eigenvectors().col(0);

	VectorXd get_v=a(0).real()*V[0];
	for (int i=1;i<V.size();i++)
		get_v+=a(i).real()*V[i];
	return get_v.dot(zn)/get_v.norm();
}

double eigens_lanczos(MatrixXd & A, VectorXd v0, VectorXd & zn, int & flops)
{
    int n=A.cols();
	double time=0;
	double st=omp_get_wtime();
    vector<double> alphas;
	vector<double> betas;
	vector<VectorXd> V;
	VectorXd v1=v0/v0.norm();
	VectorXd f = A*v1;
	double alpha = v1.dot(f);
	f = f - v1*alpha;
    V.push_back(v1);

	alphas.push_back(alpha);

    flops=n+3;
    for (int i=1;i<n;i++){
		betas.push_back(f.norm());//n
		v0 = v1;
		v1 = f/betas[betas.size()-1];//n
		f = A*v1 - v0*betas[betas.size()-1];//n^2+2n
		alphas.push_back(v1.dot(f));//n
		f = f - v1*alphas[alphas.size()-1];//2n
		V.push_back(v1);
		flops+=n+6;
		time+=omp_get_wtime()-st;
        double c=get_cosine(alphas,betas,V,zn,i+1);
		//cout<<'#'<<i<<':'<<c<<endl;
		if (abs(c)>0.999999)
			break;
		st=omp_get_wtime();
	}
	st=omp_get_wtime();
	get_cosine(alphas,betas,V,zn,alphas.size());
	flops+=V.size()+pow(V.size(),3)/n;
	time+=omp_get_wtime()-st;
	return time;
}

double eigens_vrpca(MatrixXd & X, VectorXd v0, VectorXd & zn, int & flops)
{
	int n=X.cols();
	double time=0;
	double st=omp_get_wtime();
	int m=n;
	VectorXd tmp(n);
	for (int i=0;i<n;++i){
		tmp[i]=X.col(i).dot(X.col(i));
	}
	double eta=1/(tmp.mean()*sqrt(n));
	VectorXd w=v0/v0.norm();
	VectorXd wrun=w;
	flops=2*n+1;
	for (int i=0;i<10000;i++){
		VectorXd u=X*(X.transpose()*w)/n;//2n^2
		for (int j=0;j<m;j++){
			int s=rand()%n;
			wrun+=eta*(X.col(s).dot(wrun-w)*X.col(s)+u);//6n
			wrun/=wrun.norm();//2n
		}
		w=wrun;
		time+=omp_get_wtime()-st;
		flops+=10*n;
		//cout<<w.dot(zn)<<' ';
		if (abs(w.dot(zn))>0.999999){
			break;
		}
		st=omp_get_wtime();
	}
	return time;
}

double eigens_gcd(MatrixXd & A, VectorXd x, int k, VectorXd &zn, int & flops)
{
	int n=A.cols();
	double time=0;
	flops=n;
	double st=omp_get_wtime();
	double xTx=pow(x.norm(),2);
	VectorXd Ax=A*x;
	double * wx=new double[n];
	for (int i=0;i<n;i++)
		wx[i]=abs(x[i]-Ax[i]/xTx);
	double * tmp=new double[n];
	int iter;
	for (iter=0; iter<n*n/k; iter++)
	{
		////////update X
		memcpy(tmp,wx,sizeof(double)*n);
	nth_element(tmp,tmp+n-k,tmp+n);
	double pivot=tmp[n-k];
	   for (int i=0;i<n;i++)
	   {
		   if (wx[i]>=pivot)
		   {
			   //flops++;
//			   xTx-=x[i]*x[i];
			   double oldx=x[i];
			   getnewx(A, xTx, x, Ax, i);
			   double dx=x[i]-oldx;
			   Ax+=A.col(i)*dx;
//			   xTx+=x[i]*x[i];
		   }
	   }
	   //xTx=x.dot(x);
	   //nextY = XtX.selfadjointView<Eigen::Upper>().llt().solve(AX.transpose()).transpose();
	   for (int i=0;i<n;i++)
		wx[i]=abs(x[i]-Ax[i]/xTx);
	    time+=double(omp_get_wtime()-st);
	    flops+=2+k;
		double f=Ax.dot(zn)/Ax.norm();
	    if (abs(f)>=0.999999)
			break;
	   st=omp_get_wtime();
	}
  //  cout<<iter*k<<"  Greedy coordinate descent time: ";
//	std::cout << std::fixed << std::setw( 11 ) << std::setprecision( 8 )
 //         << std::setfill( '0' ) << time/CLOCKS_PER_SEC<<endl;
	return time;
}

void initialization(MatrixXd & A, MatrixXd & Q, MatrixXd &Diagonal, int n, double lambda, VectorXd & zn)
{
	MatrixXd B=MatrixXd::Random(n,n);
//	for (int i=n/50;i<n;i++)
//		B(i,0)=0;
	 HouseholderQR<MatrixXd> qr(B);
	 Q=qr.householderQ();
	Diagonal.setZero(n,n);
	for (int i=0;i<n;i++)
		Diagonal(i,i)=(double) rand() / (RAND_MAX);
	Diagonal(0,0)=lambda;
	Diagonal(1,1)=1;
	A=Q*Diagonal*Q.transpose();
	zn=Q.col(0);
}

int main(int argc, char* argv[]){

	if (argc!=3){
		cout<<"usage: "<<argv[0]<<" <matrix size> <lambda1/lambda2>"<<endl;
		exit(0);
	}
	int n=atoi(argv[1]);
	double lambda=atof(argv[2]);
	if (lambda<=1)
	{
		cout<<"Ratio should be larger than 1!";
		exit(0);
	}

	MatrixXd A(n,n);
	VectorXd zn(n,1);
	MatrixXd Q(n,n);
	MatrixXd Diagonal(n,n);
	VectorXd x(n,1);
	VectorXd y(n,1);
	x.setRandom(n,1);
	int t=10;
	initialization(A,Q,Diagonal,n,2,zn);
	cout<<"Done with initialization"<<endl;
	//for (double lambda=0.01;lambda<=0.8;lambda+=0.02)
	//{
		Diagonal(0,0)=lambda;
		A=Q*Diagonal*Q.transpose();
		int flops=0;
		int flop;
		double flop1=0;
		double flop2=0;
		double flop3=0;
		double flop4=0;
		double timegpm=0;
		double timegcd=0;
		double timelan=0;
		double timepm=0;
		double timepca=0;
		MatrixXd X=Q*Diagonal.cwiseSqrt();
	    cout<<"Power method:";
		for (int i=0;i<t;i++){
			x.setRandom(n,1);
			timepm+=eigens_pm(A, x,zn,flop);
			flops+=flop;
		}
		flops/=t;timepm/=t;
		cout<<endl<< std::fixed << std::setprecision( 8 )
          <<timepm<<" seconds, and "<<flops<<" flops until convergence"<<endl;
		cout<<"CPM:";
		for (int i=0;i<t;i++){
			x.setRandom(n,1);
			timegpm+=eigens_gpm(A, x,n/20 ,zn,flop);
			flop1+=flop;
		}
		flop1/=t;timegpm/=t;
		std::cout << endl<< std::setprecision( 8 )
          << timegpm<<" seconds, and "<<flop1<<" flops until convergence"<<endl;
		cout<<"SGCD:";
		for (int i=0;i<t;i++){
			x.setRandom(n,1);
			timegcd+=eigens_gcd(A, x, n/20,zn,flop);
			flop2+=flop;
		}
		flop2/=t;timegcd/=t;
		std::cout << endl<< std::fixed << std::setprecision( 8 )
          << timegcd<<" seconds, and "<<flop2<<" flops until convergence"<<endl;
		cout<<"Lanczos:";
		for (int i=0;i<t;i++){
			x.setRandom(n,1);
			timelan+=eigens_lanczos(A,x,zn,flop);
			flop3+=flop;
		}
		flop3/=t;timelan/=t;
		std::cout << endl<<std::fixed << std::setprecision( 8 )
          << timelan<<" seconds, and "<<flop3<<" flops until convergence"<<endl;
		cout<<"VRPCA:";
		for (int i=0;i<t;i++){
			timepca+=eigens_vrpca(A,x,zn,flop);
			flop4+=flop;
		}
		cout<<'\n';
		flop4/=t;timepca/=t;
		std::cout << std::fixed << std::setprecision( 8 )
          <<timepca<<" seconds, and "<<flop4<<" flops until convergence"<<endl;
    /////////////////////////////////////
	//}
}
