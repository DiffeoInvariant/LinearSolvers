#ifndef JACOBI_H
#define JACOBI_H
/**
 @author: Zane Jakobs
 @brief: Jacobi iterative solver to solve diagonal-dominant system
 Ax = b
 */
#ifndef NO_MKL
#define EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include <Eigen/Core>
#include <omp.h>

namespace JacobiSolver
{
    
enum JacobiError
{
    Success                         = 0,
    NonSquareMatrixError            = 1,
    IncompatibleMatrixVectorSize    = 2,
    NonConvergenceError             = 3
    
};



using namespace Eigen;
//eventially relax the T = double restriction
template <class T = double>
class JacobiSolver
{
protected:
    /* yes, known-size at compile time would be better but this
     code should be easily portable to Python and this won't affect the speed
     too much anyway. Where it may, we'll use the known N.*/
    MatrixXd A;
    VectorXd b;
    size_t   N;//size of b (equivalently num columns of A)
    VectorXd xstar = VectorXd::Zero(N);
    size_t   NumIter;//how many iterations did the solver take?
public:
    constexpr JacobiSolver() { N = 1; };
    
    JacobiSolver(MatrixXd _A, VectorXd _b,
                     size_t _N) : A(_A), b(_b), N(_N),
                     xstar(VectorXd::Zero(N)) {};
    
    JacobiSolver(MatrixXd _A, VectorXd _b) : A(_A), b(_b)
    {
        N = A.cols();
        xstar =  VectorXd::Zero(N);
    }
    
    VectorXd getXStar() { return xstar; }
    
    size_t getNumIter() { return NumIter; }
    
    VectorXd getB() const noexcept
    {
        return b;
        
    }
    
    MatrixXd getA() const noexcept
    {
        return A;
        
    }
    size_t getN() { return N; }
    
    void setN(size_t _N){
        N = _N;
    }
    void setA(MatrixXd& _A){
        A = _A;
    }
    
    void setB(VectorXd& _b){
        b = _b;
    }
    
    void setXStar(VectorXd& _xstar){
        xstar = _xstar;
    }
    
    //make sure sizes are okay
    JacobiError CheckSizes() const
    {
        JacobiError ierr;

        size_t nAc = A.cols();
        size_t nAr = A.rows();
        size_t nB = b.size();
        
        if(nAc != nAr ) {
            ierr = NonSquareMatrixError;
            return ierr;
        }
        
        if(nAc != nB) {
            ierr = IncompatibleMatrixVectorSize;
            return ierr;
        }
        
        if(nB != N){
            N = nB;
        }
        ierr = Success;
        return ierr;
    }
    
    JacobiError Solve(size_t maxIter = 1E5, double tolerance = 1.0e-12)
    {
        /*
         iteratively solves Ax=b with an initial guess of zero, and the algorithm
         x_{k+1} = D^{-1} (b-Rx_k),
         where D are the diagonal elements of A and R the off-diagonal elements of A.
         */
        JacobiError ierr = CheckSizes();
        if(ierr > 0) { return ierr; }
        /*get R */
        MatrixXd R(N,N);
        
        double invDiag[N];
        #pragma omp parallel
        #pragma omp for
        for(size_t i =0; i < N; i++){
            for(size_t j = 0; j < N; j++){
                
                if(i == j){
                    R(i,i) = 0;
                    invDiag[i] = 1/A(i,i);
                } else{
                    R(i,j) = A(i,j);
                }
            }
        }
        
        
        VectorXd xNew(N);
        bool converged = false;
        /* main for loop*/
        for(size_t i = 0; i < maxIter && not converged; i++){
            /*should use xAXPY here, hope compiler does*/
            #pragma omp parallel
            xNew = b - R*xstar;
            #pragma omp parallel for
            for(size_t j = 0; j < N; j++){
                xNew[i] *= invDiag[i];
            }
            auto resid = xstar - xNew;
            xstar = xNew;
            //check tolerance
            if( resid.norm() < tolerance && i > 1){
                ierr = Success;
                NumIter = i;
                converged = true;
            }
            if( i == maxIter - 1 ){
                ierr = NonConvergenceError;
            }
        }
        return ierr;
    }
    
    
    
};
}

#endif
