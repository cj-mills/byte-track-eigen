#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "KalmanBBoxTrack.h"
#include "Export.h"

/**
 * Calculates the Intersection over Union (IoU) for pairs of bounding boxes.
 *
 * This function computes the IoU for each pair of boxes in two sets. The IoU is a measure of the
 * overlap between two bounding boxes.
 *
 * @param track_boxes Matrix of tracking boxes (N, 4) where N is the number of track boxes.
 * @param detection_boxes Matrix of detection boxes (M, 4) where M is the number of detection boxes.
 * @return Matrix of IoU values of size (N, M).
 */
Eigen::MatrixXd box_iou_batch(const Eigen::MatrixXd& track_boxes, const Eigen::MatrixXd& detection_boxes);

/**
 * Computes the IoU-based distance matrix for tracking purposes.
 *
 * This function converts the tracking and detection data into bounding box format and then
 * calculates the IoU matrix. The IoU matrix is then transformed into a cost matrix for tracking.
 *
 * @param track_list_a Vector of KalmanBBoxTrack, representing the first set of tracks.
 * @param track_list_b Vector of KalmanBBoxTrack, representing the second set of tracks.
 * @return A matrix representing the cost of matching tracks in track_list_a to track_list_b.
 */
Eigen::MatrixXd iou_distance(
    const std::vector<KalmanBBoxTrack>& track_list_a,
    const std::vector<KalmanBBoxTrack>& track_list_b
);

/**
 * Matches detections to tracks based on the highest IoU.
 *
 * This function calculates the IoU for each detection-track pair and assigns detections to tracks
 * based on the highest IoU value. It updates the track IDs accordingly.
 *
 * @param tlbr_boxes Matrix of bounding boxes for detections (N, 4).
 * @param tracks Vector of KalmanBBoxTrack representing the current tracks.
 * @return Vector of updated track IDs after matching.
 */
BYTE_TRACK_EIGEN_API std::vector<int> match_detections_with_tracks(
    const Eigen::MatrixXd& tlbr_boxes,
    const std::vector<KalmanBBoxTrack>& tracks
);
