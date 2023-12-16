#include "KalmanBBoxTrack.h"

// Initializing the static shared Kalman filter
KalmanFilter KalmanBBoxTrack::shared_kalman;

/**
 * @brief Default constructor for KalmanBBoxTrack.
 *        Initializes member variables to their default state.
 */
KalmanBBoxTrack::KalmanBBoxTrack() : BaseTrack(),
_tlwh(Eigen::Vector4d::Zero()), // Initializes the bounding box to a zero vector
kalman_filter(KalmanFilter()),  // Default initialization of the Kalman filter
mean(Eigen::VectorXd()),        // Initializes the mean state vector to default
covariance(Eigen::MatrixXd()),  // Initializes the covariance matrix to default
tracklet_len(0)                 // Initializes the tracklet length to zero
{ }

/**
 * @brief Constructor for KalmanBBoxTrack with initial bounding box and detection score.
 *        Throws an exception if the tlwh vector does not contain exactly 4 values.
 *
 * @param tlwh Initial bounding box in top-left width-height format.
 * @param score Detection score associated with the bounding box.
 */

KalmanBBoxTrack::KalmanBBoxTrack(const std::vector<float> tlwh, float score) :
    BaseTrack(score),
    _tlwh(tlwh.size() == 4 ? Eigen::Vector4d(tlwh[0], tlwh[1], tlwh[2], tlwh[3]) : Eigen::Vector4d::Zero()), // Ensures the bounding box has 4 elements
    kalman_filter(KalmanFilter()), // Initializes the Kalman filter
    mean(Eigen::VectorXd()),       // Initializes the mean state vector
    covariance(Eigen::MatrixXd()), // Initializes the covariance matrix
    tracklet_len(0)                // Initializes the tracklet length to zero
{
    // Validation of tlwh size
    if (tlwh.size() != 4) {
        throw std::invalid_argument("tlwh vector must contain exactly 4 values.");
    }
}

/**
 * @brief Static method to perform the prediction step on multiple KalmanBBoxTrack objects.
 *        Updates the mean and covariance of each track based on the shared Kalman filter.
 *
 * @param tracks Vector of shared pointers to KalmanBBoxTrack instances.
 */
void KalmanBBoxTrack::multi_predict(std::vector<std::shared_ptr<KalmanBBoxTrack>>& tracks) {
    if (tracks.empty()) {
        return; // Early exit if no tracks to process
    }

    // Extract mean and covariance for each track
    std::vector<Eigen::VectorXd> multi_means;
    std::vector<Eigen::MatrixXd> multi_covariances;

    for (const auto track : tracks) {
        multi_means.push_back(track->mean);  // Eigen performs deep copy by default
        multi_covariances.push_back(track->covariance);
    }

    // For each track, set velocity to 0 if it's not in Tracked state
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (tracks[i]->get_state() != TrackState::Tracked) {
            multi_means[i](7) = 0;  // Zero out the velocity if not tracked
        }
    }

    // Convert vectors to Eigen matrices for Kalman filter processing
    Eigen::MatrixXd means_matrix(multi_means.size(), multi_means[0].size());
    for (size_t i = 0; i < multi_means.size(); ++i) {
        means_matrix.row(i) = multi_means[i];
    }
    int n = (int)multi_covariances[0].rows();  // Assuming square matrices for covariance
    Eigen::MatrixXd covariances_matrix(n, n * multi_covariances.size());
    for (size_t i = 0; i < multi_covariances.size(); ++i) {
        covariances_matrix.middleCols(i * n, n) = multi_covariances[i];
    }

    // Use shared kalman filter for prediction
    Eigen::MatrixXd predicted_means, predicted_covariances;
    std::tie(predicted_means, predicted_covariances) = shared_kalman.multi_predict(means_matrix, covariances_matrix);

    // Update each track with the predicted mean and covariance
    for (size_t i = 0; i < tracks.size(); ++i) {
        tracks[i]->mean = predicted_means.col(i);
    }

    int pcr = (int)predicted_covariances.rows();  // Assuming the block matrix is composed of square matrices
    size_t num_matrices = predicted_covariances.cols() / pcr;

    for (size_t i = 0; i < num_matrices; ++i) {
        Eigen::MatrixXd block = predicted_covariances.middleCols(i * pcr, pcr);
        tracks[i]->covariance = block;
    }
}

/**
 * @brief Converts bounding box from tlwh format to xyah format (center x, center y, aspect ratio, height).
 *
 * @param tlwh Bounding box in tlwh format.
 * @return Eigen::VectorXd Bounding box in xyah format.
 */
Eigen::VectorXd KalmanBBoxTrack::tlwh_to_xyah(Eigen::VectorXd tlwh) {
    Eigen::VectorXd ret = tlwh;
    ret.head(2) += ret.segment(2, 2) / 2.0; // Adjust x, y to be the center of the bounding box
    ret(2) /= ret(3); // Update aspect ratio
    return ret;
}

/**
 * @brief Activates the track with a given Kalman filter and frame ID.
 *        Initializes the track ID and sets up the Kalman filter with the bounding box information.
 *
 * @param kalman_filter Kalman filter to be used for this track.
 * @param frame_id Frame ID at which this track is activated.
 */
void KalmanBBoxTrack::activate(KalmanFilter& kalman_filter, int frame_id) {
    // Link to the provided Kalman filter
    this->kalman_filter = kalman_filter;

    // Initialize track ID
    this->track_id = BaseTrack::next_id();

    // Convert bounding box to the xyah format and initiate the Kalman filter
    std::tie(this->mean, this->covariance) = this->kalman_filter.initiate(this->tlwh_to_xyah(this->_tlwh));

    this->tracklet_len = 0;
    this->state = TrackState::Tracked;

    // Check if track is activated in the first frame
    this->is_activated = (frame_id == 1);
    this->frame_id = frame_id;
    this->start_frame = frame_id;
}

/**
 * @brief Updates the track with a new detection, potentially assigning a new ID.
 *
 * @param new_track The new detection represented as a KalmanBBoxTrack object.
 * @param frame_id Frame ID of the new detection.
 * @param new_id Flag indicating whether to assign a new ID to this track.
 */
void KalmanBBoxTrack::update_track(const KalmanBBoxTrack& new_track, int frame_id, bool new_id) {
    // Track update logic
    this->frame_id = frame_id;
    this->tracklet_len++;
    
    std::tie(this->mean, this->covariance) = this->kalman_filter.update(this->mean, this->covariance, this->tlwh_to_xyah(new_track.tlwh()));

    this->state = TrackState::Tracked;
    this->is_activated = true;

    // Assigning a new ID if necessary
    if (new_id) {
        this->track_id = BaseTrack::next_id();
    }

    // Update the score
    this->score = new_track.get_score();
}

/**
 * @brief Re-activates the track with a new detection, possibly assigning a new ID.
 *        Essentially a wrapper for the update_track method.
 *
 * @param new_track The new detection to reactivate the track with.
 * @param frame_id Frame ID of the reactivation.
 * @param new_id Flag indicating whether to assign a new ID.
 */
void KalmanBBoxTrack::re_activate(const KalmanBBoxTrack& new_track, int frame_id, bool new_id) {
    update_track(new_track, frame_id, new_id);
}

/**
 * @brief Updates the track with a new detection without changing its ID.
 *        Wrapper for the update_track method without the new_id flag.
 *
 * @param new_track The new detection to update the track with.
 * @param frame_id Frame ID of the update.
 */
void KalmanBBoxTrack::update(const KalmanBBoxTrack& new_track, int frame_id) {
    update_track(new_track, frame_id);
}

/**
 * @brief Converts tlwh bounding box to top-left bottom-right (tlbr) format.
 *
 * @param tlwh Bounding box in tlwh format.
 * @return Eigen::Vector4d Bounding box in tlbr format.
 */
Eigen::Vector4d KalmanBBoxTrack::tlwh_to_tlbr(const Eigen::Vector4d tlwh) {
    Eigen::Vector4d ret = tlwh;
    ret.tail<2>() += ret.head<2>();
    return ret;
}

/**
 * @brief Converts tlbr bounding box to tlwh format.
 *
 * @param tlbr Bounding box in tlbr format.
 * @return Eigen::Vector4d Bounding box in tlwh format.
 */
Eigen::Vector4d KalmanBBoxTrack::tlbr_to_tlwh(const Eigen::Vector4d tlbr) {
    Eigen::Vector4d ret = tlbr;
    ret.tail<2>() -= ret.head<2>();
    return ret;
}

/**
 * @brief Get current bounding box in tlwh format
 *
 * @return Eigen::Vector4d Bounding box in tlwh format.
 */
Eigen::Vector4d KalmanBBoxTrack::tlwh() const {
    if (mean.isZero(0)) { // Checking if 'mean' is uninitialized or zero
        return _tlwh;
    }

    Eigen::Vector4d ret = mean.head(4);
    ret[2] *= ret[3];
    ret[0] -= ret[2] / 2.0;
    ret[1] -= ret[3] / 2.0;

    return ret;
}

/**
 * @brief Returns the bounding box in tlbr format.
 *
 * @return Eigen::Vector4d Bounding box in tlbr format.
 */
Eigen::Vector4d KalmanBBoxTrack::tlbr() const {
    return this->tlwh_to_tlbr(this->tlwh());
}