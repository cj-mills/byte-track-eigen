#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include <stdexcept>
#include "Export.h"

/// <summary>
/// KalmanFilter is a class designed for tracking bounding boxes in image space.
/// It uses a state estimation technique that employs a series of measurements observed over time,
/// containing statistical noise and other inaccuracies, and produces estimates of unknown variables
/// that tend to be more precise than those based on a single measurement alone.
/// </summary>
class BYTE_TRACK_EIGEN_API KalmanFilter
{
public:
    /// Constructor for KalmanFilter.
    KalmanFilter();

    /// Destructor for KalmanFilter.
    ~KalmanFilter();

    /// Compute standard deviations based on the mean.
    /// This is used for initializing the uncertainty in the filter.
    /// @param mean The mean vector from which standard deviations are computed.
    /// @return Vector of standard deviations.
    Eigen::VectorXd create_std(const Eigen::VectorXd& mean);

    /// Initialize a new track from an unassociated measurement.
    /// This method is typically called to create a new track with an initial measurement.
    /// @param measurement The initial measurement vector for the track.
    /// @return A pair of mean and covariance matrix representing the initial state estimate.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> initiate(const Eigen::VectorXd& measurement);

    /// Project the state distribution to the measurement space.
    /// This method is used to predict the next state of the object based on the current state.
    /// @param mean The current state mean vector.
    /// @param covariance The current state covariance matrix.
    /// @return A pair of mean and covariance matrix representing the predicted state.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> project(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance
    );

    /// Run the Kalman filter prediction step for multiple measurements.
    /// This method is used when multiple measurements are available simultaneously.
    /// @param mean The mean matrix of all current states.
    /// @param covariance The covariance matrix of all current states.
    /// @return A pair of mean and covariance matrices representing the predicted states.
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> multi_predict(
        const Eigen::MatrixXd& mean,
        const Eigen::MatrixXd& covariance
    );

    /// Run the Kalman filter correction step.
    /// This method updates the state of the object based on the received measurement.
    /// @param mean The predicted state mean vector.
    /// @param covariance The predicted state covariance matrix.
    /// @param measurement The new measurement vector.
    /// @return A pair of mean and covariance matrix representing the updated state.
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance,
        const Eigen::VectorXd& measurement
    );

private:
    int ndim; // The dimension of the state space.
    Eigen::MatrixXd motion_mat; // The motion model matrix, used to predict the next state.
    Eigen::MatrixXd update_mat; // The update matrix used for projecting state distribution to measurement space.
    double std_weight_position; // Standard deviation weight for the position, used in uncertainty modeling.
    double std_weight_velocity; // Standard deviation weight for the velocity, used in uncertainty modeling.
};
