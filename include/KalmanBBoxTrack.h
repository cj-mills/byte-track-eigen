#pragma once

#include <iostream>
#include "BaseTrack.h"
#include "KalmanFilter.h"
#include <vector>
#include <Eigen/Dense>
#include <memory>

/**
 * @brief The KalmanBBoxTrack class extends the BaseTrack class and incorporates a Kalman filter for object tracking.
 *        It specializes in tracking objects represented by bounding boxes in the format of top-left corner coordinates,
 *        width, and height (tlwh format).
 */
class KalmanBBoxTrack : public BaseTrack {
private:
    static KalmanFilter shared_kalman;  // A shared Kalman filter instance for multi-object tracking purposes.

public:
    Eigen::Vector4d _tlwh;              // Bounding box of the tracked object in top-left width-height (tlwh) format.
    KalmanFilter kalman_filter;         // An individual Kalman filter instance for this specific track.
    Eigen::VectorXd mean;               // The mean state vector of the Kalman filter, representing the tracked object's state.
    Eigen::MatrixXd covariance;         // The covariance matrix of the Kalman filter state, representing the uncertainty in the tracked state.
    int tracklet_len;                   // The length of the tracklet, indicating the number of consecutive frames where the object has been tracked.

    /**
     * @brief Default constructor for KalmanBBoxTrack.
     */
    KalmanBBoxTrack();

    /**
     * @brief Constructor for KalmanBBoxTrack with initial bounding box and detection score.
     *
     * @param tlwh Initial bounding box in top-left width-height format.
     * @param score Detection score associated with the bounding box.
     */
    KalmanBBoxTrack(const std::vector<float> tlwh, float score);

    /**
     * @brief Static method to perform the prediction step on multiple KalmanBBoxTrack objects.
     *        Updates the mean and covariance of each track based on the shared Kalman filter.
     *
     * @param tracks Vector of shared pointers to KalmanBBoxTrack instances.
     */
    static void multi_predict(std::vector<std::shared_ptr<KalmanBBoxTrack>>& tracks);

    /**
     * @brief Converts bounding box from tlwh format to xyah format (center x, center y, aspect ratio, height).
     *
     * @param tlwh Bounding box in tlwh format.
     * @return Eigen::VectorXd Bounding box in xyah format.
     */
    Eigen::VectorXd tlwh_to_xyah(Eigen::VectorXd tlwh);

    /**
     * @brief Activates the track with a given Kalman filter and frame ID.
     *        Initializes the track ID and sets up the Kalman filter with the bounding box information.
     *
     * @param kalman_filter Kalman filter to be used for this track.
     * @param frame_id Frame ID at which this track is activated.
     */
    void activate(KalmanFilter& kalman_filter, int frame_id);

    /**
     * @brief Updates the track with a new detection, potentially assigning a new ID.
     *
     * @param new_track The new detection represented as a KalmanBBoxTrack object.
     * @param frame_id Frame ID of the new detection.
     * @param new_id Flag indicating whether to assign a new ID to this track.
     */
    void update_track(const KalmanBBoxTrack& new_track, int frame_id, bool new_id = false);

    /**
     * @brief Re-activates the track with a new detection, possibly assigning a new ID.
     *        Essentially a wrapper for the update_track method.
     *
     * @param new_track The new detection to reactivate the track with.
     * @param frame_id Frame ID of the reactivation.
     * @param new_id Flag indicating whether to assign a new ID.
     */
    void re_activate(const KalmanBBoxTrack& new_track, int frame_id, bool new_id = false);

    /**
     * @brief Updates the track with a new detection without changing its ID.
     *
     * @param new_track The new detection to update the track with.
     * @param frame_id Frame ID of the update.
     */
    void update(const KalmanBBoxTrack& new_track, int frame_id);

    /**
     * @brief Converts tlwh bounding box to top-left bottom-right (tlbr) format.
     *
     * @param tlwh Bounding box in tlwh format.
     * @return Eigen::Vector4d Bounding box in tlbr format.
     */
    static Eigen::Vector4d tlwh_to_tlbr(const Eigen::Vector4d tlwh);

    /**
     * @brief Converts tlbr bounding box to tlwh format.
     *
     * @param tlbr Bounding box in tlbr format.
     * @return Eigen::Vector4d Bounding box in tlwh format.
     */
    static Eigen::Vector4d tlbr_to_tlwh(const Eigen::Vector4d tlbr);

    /**
     * @brief Returns the bounding box in tlwh format.
     *
     * @return Eigen::Vector4d Bounding box in tlwh format.
     */
    Eigen::Vector4d tlwh() const;

    /**
     * @brief Returns the bounding box in tlbr format.
     *
     * @return Eigen::Vector4d Bounding box in tlbr format.
     */
    Eigen::Vector4d tlbr() const;
};
