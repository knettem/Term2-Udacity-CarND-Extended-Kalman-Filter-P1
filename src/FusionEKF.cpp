#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;
    previous_timestamp_ = 0;
    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.9, 0, 0,
            0, 0.009, 0,
            0, 0, 0.9;

    // Measurement matrix H_laser_ for mapping state x = [px, py, vx, vy] to the measured [px, py] vector
    H_laser_ << 1, 0, 0, 0,
                  0, 1, 0, 0;

    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            1, 0, 1, 0,
            0, 1, 0, 1;

    //state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

    //the initial state transition matrix
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

}
/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];

            ekf_.x_ << rho * cos(phi), rho * sin(phi), 0, 0;
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
             Initialize state.
             */
            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            ekf_.x_ << rho, phi, 0, 0;
        }
        previous_timestamp_ = measurement_pack.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

      float dt_2 = dt * dt;
      float dt_3 = dt_2 * dt;
      float dt_4 = dt_3 * dt;

      //Modify the F matrix so that the time is integrated
      ekf_.F_(0, 2) = dt;
      ekf_.F_(1, 3) = dt;

      //process noise
      float noise_ax = 9;
      float noise_ay = 9;

      //set the process covariance matrix Q
      ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
                0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
                dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
                0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

      //predict
      ekf_.Predict();
    /*****************************************************************************
     *  Update
     ****************************************************************************/

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        std::cout << "RADAR" << std::endl;
        // Radar updates
        Hj_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.H_ = Hj_;
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        //Laser updates
        std::cout << "LASER" << std::endl;
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;

        ekf_.Update(measurement_pack.raw_measurements_);
    }
    // print the output
    cout << "After update x_ = " << ekf_.x_ << endl;
    cout << "After update P_ = " << ekf_.P_ << endl;
}
