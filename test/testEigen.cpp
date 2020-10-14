#include <Eigen/Sparse>
#include <iostream>

int main(){
    Eigen::SparseMatrix<double> m1(3,3);
    m1.coeffRef(0,2) = 3.0;
    m1.coeffRef(0,1) = 2.0;
    std::cout << m1.coeffRef(0,1) << std::endl;
    m1.coeffRef(0,1) -= 1.0;

    std::cout << m1 << std::endl;
    return 0;
}