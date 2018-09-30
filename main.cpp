#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

int main() {
    //初期条件
    const int nstep = 4000;
    const int n = 100;   //格子点数
    const double dx = 0.1;
    const double dt = 0.0005;
    const double gamma = 1.4;

    Matrix<double, n, 3> q0;    //保存量
    Matrix<double, n, 3> q_old;    //保存量
    //Matrix<double, n, 3> q_mean;    //保存量
    Matrix<double, n, 3> q_new;    //保存量
    Matrix<double, n, 3> e0;    //フラックス
    Matrix<double, n, 3> e_old;    //フラックス
    Matrix<double, n, 3> e_new;    //フラックス
    Matrix<double, n, 3> e_mean;    //j+1/2の値をjに格納しているフラックス

    //左側の要素が大きな値
    const double rho0_l = 2.0;
    const double u0_l = 0;
    const double en0_l = 2.0;
    //右側の要素が小さな値
    const double rho0_r = 1.0;
    const double u0_r = 0;
    const double en0_r = 1.0;

    q0 = MatrixXd::Zero(n,3);
    for(int i=0;i< static_cast<int>(n/2);i++){
        q0(i,0)=rho0_l;
        q0(i,1)=rho0_l*u0_l;
        q0(i,2)=en0_l;
    }
    for(int i=static_cast<int>(n/2);i<n;i++){
        q0(i,0)=rho0_r;
        q0(i,1)=rho0_r*u0_r;
        q0(i,2)=en0_r;
    }

    e0 = MatrixXd::Zero(n,3);
    for(int i=0;i< static_cast<int>(n/2);i++){
        e0(i,0)=rho0_l*u0_l;
        e0(i,1)=(gamma-1)*en0_l+0.5*(3-gamma)*rho0_l*u0_l*u0_l;
        e0(i,2)=gamma*en0_l*u0_l - 0.5*(gamma-1)*rho0_l*pow(u0_l,3.0);
    }
    for(int i=static_cast<int>(n/2);i<n;i++){
        e0(i,0)=rho0_r*u0_r;
        e0(i,1)=(gamma-1)*en0_r+0.5*(3-gamma)*rho0_r*u0_r*u0_r;
        e0(i,2)=gamma*en0_r*u0_r - 0.5*(gamma-1)*rho0_r*pow(u0_r,3.0);
    }
    q_old = q0;
    e_old = e0;

    //初期条件設定終わり

    //A_j+1/2を求めるためにRoeの平均計算する
    //j=0,nの点は値を保存
    //なので，計算は1~n-1までの格子点で行う
    Matrix<double, 3, 3> A_abs;
    Matrix<double, 3, 3> par_A;
    Matrix<double, 3, 3> R;
    Matrix<double, 3, 3> R_inv;
    Matrix<double, 3, 3> Gam;
    double rho_ave;
    double u_ave;
    double H_ave;
    double c_ave;
    double D;   //D=sqrt(rho[1]/rho[0])
    double p[2];
    double H[2];
    double rho[2];
    double m[2];
    double e[2];
    //計算するときのパラメタ
    double a_par;
    double b_par;

    //時間ステップだけ行う
    for (int i=0; i<nstep; i++) {

        for (int j = 0; j < n - 1; j++) {
            for (int k = 0; k < 2; k++) {
                rho[k] = q_old(j + k, 0);
                m[k] = q_old(j + k, 1);
                e[k] = q_old(j + k, 2);
            }
            //まずは，保存量からp,Hを計算
            for (int k = 0; k < 2; k++) {
                p[k] = (gamma - 1) * (e[k] - 0.5 * rho[k] * pow(m[k] / rho[k], 2));
            }
            for (int k = 0; k < 2; k++) {
                H[k] = (e[k] + p[k]) / rho[k];
            }

            //Roe average
            D = sqrt(rho[1] / rho[0]);   //パラメタ
            rho_ave = sqrt(rho[0] * rho[1]);
            u_ave = (m[0] / rho[0] + D * m[1] / rho[1]) / (1 + D);
            H_ave = (H[0] + D * H[1]) / (1 + D);
            c_ave = sqrt((gamma - 1) * (H_ave - 0.5 * pow(u_ave, 2)));

            //一次精度風上差分
            b_par = (gamma - 1) / pow(c_ave, 2);
            a_par = 0.5 * (b_par * pow(u_ave, 2));

            R << 1, 1, 1,
                    u_ave - c_ave, u_ave, u_ave + c_ave,
                    H_ave - u_ave * c_ave, 0.5 * pow(u_ave, 2), H_ave + u_ave * c_ave;

            R_inv << 0.5 * (a_par + u_ave / c_ave), 0.5 * (-b_par * u_ave - 1 / c_ave), 0.5 * b_par,
                    1 - a_par, b_par * u_ave, -b_par,
                    0.5 * (a_par - u_ave / c_ave), 0.5 * (-b_par * u_ave + 1 / c_ave), 0.5 * b_par;

            Gam << std::abs(u_ave - c_ave), 0, 0,
                    0, std::abs(u_ave), 0,
                    0, 0, std::abs(u_ave + c_ave);

            par_A = R * Gam ;
            A_abs = par_A * R_inv;

            if (i==0 && j == 50){
                //std::cout << A_abs << std::endl;
                //std::cout << (q_old.row(j + 1).transpose() - q_old.row(j).transpose()) << std::endl;
                std::cout << R*R_inv << std::endl;
            }
            //std::cout << A_abs << std::endl;

            //j+1/2のフラックスの計算
            e_mean.row(j).transpose() = 0.5 * (e_old.row(j + 1).transpose() + e_old.row(j).transpose() -
                                               A_abs * (q_old.row(j + 1).transpose() - q_old.row(j).transpose()));
        }
        /*if (i==0){
            std::cout << e_mean << std::endl;
        }*/


        //次の時刻の値に更新
        //j-1/2の値が要求されるので，更新できるのは1<j<n-1まで
        double p_new;
        double rho_new;
        double u_new;
        double ene_new;

        //境界付近の保存量は普遍であるとする
        e_new.row(0) = e_old.row(0);
        e_new.row(n-1) = e_old.row(n-1);
        q_new.row(0) = q_old.row(0);
        q_new.row(n-1) = q_old.row(n-1);

        for (int j = 1; j < n - 1; j++) {
            q_new.row(j) =
                    q_old.row(j) - dt / dx * (e_mean.row(j) - e_mean.row(j - 1));

            rho_new = q_new(j, 0);
            u_new = q_new(j, 1) / q_new(j, 0);
            ene_new = q_new(j, 2);
            p_new = (gamma - 1) * (ene_new - 0.5 * rho_new * pow(u_new, 2));

            e_new(j, 0) = rho_new * u_new;
            e_new(j, 1) = p_new + rho_new * pow(u_new, 2);
            e_new(j, 2) = (ene_new + p_new) * u_new;

        }

        q_old = q_new;
        e_old = e_new;

        if (i%100 == 0){
            std::cout << i << std::endl;
        }

        if (i == nstep-1){
            std::ofstream ofs("q_500.csv");
            if (!ofs) {
                std::cerr << "ファイルオープンに失敗" << std::endl;
                std::exit(1);

            }
            for (int i=0; i<n; i++){
                ofs << dx*i << "," << q_new(i,0) << "," << q_new(i,1)/q_new(i,0) << "," <<
                    (gamma - 1) * (q_new(i,2) - 0.5 * q_new(i,0) * pow(q_new(i,1)/q_new(i,0), 2)) << std::endl;
            }

        }

    }


    return 0;
}