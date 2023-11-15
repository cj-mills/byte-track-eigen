#pragma once

#include <stdexcept>
#include <stdlib.h>
#include <memory>
#include <vector>
#include <sstream>
#include "KalmanFilter.h"
#include "KalmanBBoxTrack.h"
#include "BaseTrack.h"
#include "BoundingBoxIoUMatching.h"
#include "LinearAssignment.h"
#include "BoundingBoxTrackUtils.h"
#include "Export.h"

/**
 * @brief BYTETracker class for tracking objects in video frames using Kalman Filters.
 *
 * This class implements the BYTETrack algorithm, which combines the use of Kalman Filters
 * and Hungarian algorithm for object tracking in video streams.
 *
 * Key functionalities include processing frame detections, updating track states, and
 * maintaining lists of active, lost, and removed tracks.
 */
class BYTE_TRACK_EIGEN_API BYTETracker {
public:
    /**
     * @brief Constructor for BYTETracker.
     *
     * Initializes the tracker with specific thresholds and parameters for tracking.
     *
     * @param track_thresh Threshold for track confidence.
     * @param track_buffer Size of the buffer to store track history.
     * @param match_thresh Threshold for matching detections to tracks.
     * @param frame_rate Frame rate of the video being processed.
     */
    BYTETracker(float track_thresh = 0.25, int track_buffer = 30, float match_thresh = 0.8, int frame_rate = 30);

private:
    // Internal constants and thresholds
    const float BASE_FRAME_RATE = 30.0;
    const float MIN_KEEP_THRESH = 0.1f;
    const float LOWER_CONFIDENCE_MATCHING_THRESHOLD = 0.5;
    const float ACTIVATION_MATCHING_THRESHOLD = 0.7;

    // Member variables for tracking settings and state
    float track_thresh; // Threshold for determining track validity.
    float match_thresh; // Threshold for matching detections to existing tracks.
    int frame_id;       // Current frame identifier.
    float det_thresh;   // Detection threshold for filtering weak detections.
    int buffer_size;    // Size of the history buffer for tracks.
    int max_time_lost;  // Maximum time a track can be lost before removal.

    // Kalman filter and track lists
    KalmanFilter kalman_filter; // Kalman filter for state estimation.
    std::vector<std::shared_ptr<KalmanBBoxTrack>> tracked_tracks; // Active tracks.
    std::vector<std::shared_ptr<KalmanBBoxTrack>> lost_tracks;    // Tracks that are currently lost.
    std::vector<std::shared_ptr<KalmanBBoxTrack>> removed_tracks; // Tracks that are removed.

    LinearAssignment linear_assignment; // For solving assignment problems.

    // Internal methods for processing detections and tracks
    std::vector<KalmanBBoxTrack> extract_kalman_bbox_tracks(const Eigen::MatrixXf dets, const Eigen::VectorXf scores_keep);
    Eigen::MatrixXf select_matrix_rows_by_indices(const Eigen::MatrixXf matrix, const std::vector<int> indices);
    std::pair<std::vector<KalmanBBoxTrack>, std::vector<KalmanBBoxTrack>> filter_and_partition_detections(const Eigen::MatrixXf& output_results);
    std::pair<std::vector<std::shared_ptr<KalmanBBoxTrack>>, std::vector<std::shared_ptr<KalmanBBoxTrack>>> partition_tracks_by_activation();
    std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>> assign_tracks_to_detections(
        const std::vector<std::shared_ptr<KalmanBBoxTrack>> tracks,
        const std::vector<KalmanBBoxTrack> detections,
        double thresh
    );
    void update_tracks_from_detections(
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& tracks,
        const std::vector<KalmanBBoxTrack> detections,
        const std::vector<std::pair<int, int>> track_detection_pair_indices,
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& reacquired_tracked_tracks,
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& activated_tracks
    );
    std::vector<std::shared_ptr<KalmanBBoxTrack>> extract_active_tracks(
        const std::vector<std::shared_ptr<KalmanBBoxTrack>>& tracks,
        std::set<int> unpaired_track_indices
    );
    void flag_unpaired_tracks_as_lost(
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& currently_tracked_tracks,
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& lost_tracks,
        std::set<int> unpaired_track_indices
    );
    void prune_and_merge_tracked_tracks(
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& reacquired_tracked_tracks,
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& activated_tracks
    );
    void handle_lost_and_removed_tracks(
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& removed_tracks,
        std::vector<std::shared_ptr<KalmanBBoxTrack>>& lost_tracks
    );

public:
    /**
     * @brief Process detections for a single frame and update track states.
     *
     * This function takes the detections for a frame and updates the state of the tracks
     * by matching, creating new tracks, and updating existing ones.
     *
     * @param output_results Matrix of detection results for the current frame.
     * @return std::vector<KalmanBBoxTrack> List of updated tracks after processing the frame.
     */
    std::vector<KalmanBBoxTrack> process_frame_detections(const Eigen::MatrixXf& output_results);

};
