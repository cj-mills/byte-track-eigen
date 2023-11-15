#include "KalmanFilter.h"

/// Constructor for KalmanFilter.
/// Initializes the filter with default parameters for tracking bounding boxes in image space.
KalmanFilter::KalmanFilter() {
    ndim = 4;  // State space dimension (x, y, width, height)
    double dt = 1.0;  // Time step, assuming time in seconds

    // Initializing the motion matrix for the Kalman Filter.
    // This matrix is used to predict the next state of the tracked object.
    motion_mat = Eigen::MatrixXd::Identity(2 * ndim, 2 * ndim);
    motion_mat.block(0, ndim, ndim, ndim) = dt * Eigen::MatrixXd::Identity(ndim, ndim);

    // The update matrix is used for updating the state with a new measurement.
    update_mat = Eigen::MatrixXd::Identity(ndim, 2 * ndim);

    // Setting standard deviation weights for position and velocity.
    // These weights help model the uncertainty in the measurements.
    std_weight_position = 1. / 20.;
    std_weight_velocity = 1. / 160.;
}

/// Destructor for KalmanFilter.
KalmanFilter::~KalmanFilter() {}

/// Compute standard deviations based on the mean.
/// This function creates a standard deviation vector for the state uncertainty.
/// @param mean The mean vector of the state.
/// @return Vector of standard deviations.
Eigen::VectorXd KalmanFilter::create_std(const Eigen::VectorXd& mean) {
    Eigen::VectorXd std_devs(8);
    // Setting standard deviations for each state dimension based on the weights.
    std_devs(0) = std_weight_position * mean(3);
    std_devs(1) = std_weight_position * mean(3);
    std_devs(2) = 1e-2;  // Small fixed standard deviation for width
    std_devs(3) = std_weight_position * mean(3);
    std_devs(4) = std_weight_velocity * mean(3);
    std_devs(5) = std_weight_velocity * mean(3);
    std_devs(6) = 1e-5;  // Small fixed standard deviation for velocity in width
    std_devs(7) = std_weight_velocity * mean(3);

    return std_devs;
}

/// Initialize a new track from an unassociated measurement.
/// @param measurement The initial measurement vector for the track.
/// @return A pair of mean and covariance matrix representing the initial state estimate.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::initiate(const Eigen::VectorXd& measurement) {

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(2 * ndim);
    mean.head(ndim) = measurement;  // Initial state mean is set to the measurement

    // Calculating initial covariance matrix based on the standard deviations
    Eigen::VectorXd std_devs = create_std(mean);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(2 * ndim, 2 * ndim);
    for (int i = 0; i < 2 * ndim; i++) {
        covariance(i, i) = std_devs(i) * std_devs(i);
    }

    return { mean, covariance };
}

/// Project the state distribution to the measurement space.
/// @param mean The current state mean vector.
/// @param covariance The current state covariance matrix.
/// @return A pair of mean and covariance matrix representing the predicted state.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::project(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) {
    Eigen::VectorXd projected_std = create_std(mean).head(ndim);
    Eigen::MatrixXd innovation_cov = Eigen::MatrixXd::Zero(ndim, ndim);
    for (int i = 0; i < ndim; i++) {
        innovation_cov(i, i) = projected_std(i) * projected_std(i);
    }

    // Predicting the new mean and covariance matrix for the measurement space
    Eigen::VectorXd new_mean = this->update_mat * mean;
    Eigen::MatrixXd new_cov = this->update_mat * covariance * this->update_mat.transpose() + innovation_cov;

    return { new_mean, new_cov };
}

/// Run the Kalman filter prediction step for multiple measurements.
/// @param means The mean matrix of all current states.
/// @param covariances The covariance matrix of all current states.
/// @return A pair of mean and covariance matrices representing the predicted states.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KalmanFilter::multi_predict(const Eigen::MatrixXd& means, const Eigen::MatrixXd& covariances) {

    int n_tracks = means.rows();  // Number of tracks

    // Prepare matrices to hold new means and covariances after prediction.
    Eigen::MatrixXd new_means(2 * ndim, n_tracks);
    Eigen::MatrixXd new_covs(2 * ndim, 2 * ndim * n_tracks);  // Stacking covariances along the last two dimensions

    for (int i = 0; i < n_tracks; ++i) {
        // Calculating motion standard deviation and covariance for each track
        Eigen::VectorXd motion_std = create_std(means.row(i));
        Eigen::MatrixXd motion_cov = Eigen::MatrixXd::Zero(2 * ndim, 2 * ndim);
        for (int j = 0; j < 2 * ndim; j++) {
            motion_cov(j, j) = motion_std(j) * motion_std(j);
        }

        // Updating means and covariances for each track
        new_means.col(i) = motion_mat * means.row(i).transpose();
        new_covs.block(0, 2 * ndim * i, 2 * ndim, 2 * ndim) = motion_mat * covariances.block(0, 2 * ndim * i, 2 * ndim, 2 * ndim) * motion_mat.transpose() + motion_cov;
    }

    return { new_means, new_covs };
}

/// Run the Kalman filter correction step.
/// @param mean The predicted state mean vector.
/// @param covariance The predicted state covariance matrix.
/// @param measurement The new measurement vector.
/// @return A pair of mean and covariance matrix representing the updated state.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::update(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const Eigen::VectorXd& measurement)
{
    // Projecting the state to the measurement space
    Eigen::VectorXd projected_mean;
    Eigen::MatrixXd projected_cov;
    std::tie(projected_mean, projected_cov) = project(mean, covariance);

    // Performing Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> cho_factor(projected_cov);
    if (cho_factor.info() != Eigen::Success) {
        // Handling decomposition failure
        throw std::runtime_error("Decomposition failed!");
    }

    // Calculating the Kalman gain
    // This represents how much the predictions should be corrected based on the new measurement.
    Eigen::MatrixXd kalman_gain = cho_factor.solve(this->update_mat * covariance).transpose();

    // Updating the mean and covariance based on the new measurement and Kalman gain
    Eigen::VectorXd new_mean = mean + kalman_gain * (measurement - projected_mean);
    Eigen::MatrixXd new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();

    return { new_mean, new_covariance };
}
