#include "BoundingBoxIoUMatching.h"

/**
 * Calculates the Intersection over Union (IoU) for pairs of bounding boxes.
 *
 * @param track_boxes Matrix of tracking boxes, expected to be in the format (N, 4).
 * @param detection_boxes Matrix of detection boxes, expected to be in the format (M, 4).
 * @return Matrix of IoU values, with dimensions (N, M).
 */
Eigen::MatrixXd box_iou_batch(const Eigen::MatrixXd& track_boxes, const Eigen::MatrixXd& detection_boxes) {
    // Validate the shape of input matrices
    if (track_boxes.cols() != 4 || detection_boxes.cols() != 4) {
        throw std::invalid_argument("Input matrices must have 4 columns each.");
    }

    int N = (int)track_boxes.rows();
    int M = (int)detection_boxes.rows();

    // Calculate areas of the track boxes and detection boxes
    Eigen::VectorXd track_areas = (track_boxes.col(2) - track_boxes.col(0))
        .cwiseProduct(track_boxes.col(3) - track_boxes.col(1));
    Eigen::VectorXd detection_areas = (detection_boxes.col(2) - detection_boxes.col(0))
        .cwiseProduct(detection_boxes.col(3) - detection_boxes.col(1));

    // Initialize the IoU matrix
    Eigen::MatrixXd iou_matrix(N, M);

    // Compute IoU for each pair of boxes
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            // Calculate intersection box coordinates
            double inter_x_min = std::max(track_boxes(i, 0), detection_boxes(j, 0));
            double inter_y_min = std::max(track_boxes(i, 1), detection_boxes(j, 1));
            double inter_x_max = std::min(track_boxes(i, 2), detection_boxes(j, 2));
            double inter_y_max = std::min(track_boxes(i, 3), detection_boxes(j, 3));

            // Compute intersection area
            double inter_width = std::max(inter_x_max - inter_x_min, 0.0);
            double inter_height = std::max(inter_y_max - inter_y_min, 0.0);
            double inter_area = inter_width * inter_height;

            // Calculate IoU
            iou_matrix(i, j) = inter_area / (track_areas(i) + detection_areas(j) - inter_area);
        }
    }

    return iou_matrix;
}

/**
 * Computes the IoU-based distance matrix for tracking purposes.
 *
 * @param track_list_a Vector of KalmanBBoxTrack, representing the first set of tracks.
 * @param track_list_b Vector of KalmanBBoxTrack, representing the second set of tracks.
 * @return A matrix representing the cost of matching tracks in track_list_a to track_list_b.
 */
Eigen::MatrixXd iou_distance(
    const std::vector<KalmanBBoxTrack>& track_list_a,
    const std::vector<KalmanBBoxTrack>& track_list_b
) {
    size_t m = track_list_a.size();
    size_t n = track_list_b.size();

    // Extract bounding boxes from tracks
    Eigen::MatrixXd tlbr_list_a(m, 4);
    for (size_t i = 0; i < m; i++) {
        tlbr_list_a.row(i) = track_list_a[i].tlbr();
    }

    Eigen::MatrixXd tlbr_list_b(n, 4);
    for (size_t i = 0; i < n; i++) {
        tlbr_list_b.row(i) = track_list_b[i].tlbr();
    }

    // Calculate IoUs and form the cost matrix
    Eigen::MatrixXd ious;
    if (tlbr_list_a.rows() == 0 || tlbr_list_b.rows() == 0) {
        ious = Eigen::MatrixXd::Zero(tlbr_list_a.rows(), tlbr_list_b.rows());
    }
    else {
        ious = box_iou_batch(tlbr_list_a, tlbr_list_b);
    }
    Eigen::MatrixXd cost_matrix = Eigen::MatrixXd::Ones(m, n) - ious;

    return cost_matrix;
}

/**
 * Matches detections to tracks based on the highest IoU.
 *
 * @param tlbr_boxes Matrix of bounding boxes for detections (N, 4).
 * @param tracks Vector of KalmanBBoxTrack representing the current tracks.
 * @return Vector of updated track IDs after matching.
 */
std::vector<int> match_detections_with_tracks(
    const Eigen::MatrixXd& tlbr_boxes,
    const std::vector<KalmanBBoxTrack>& tracks
) {
    size_t m = tracks.size();
    size_t n = tlbr_boxes.rows();

    // Clone the input track_ids to operate on
    std::vector<int> track_ids(n, -1);

    // Extract bounding boxes from tracks
    Eigen::MatrixXd track_boxes(m, 4);
    for (size_t i = 0; i < m; i++) {
        track_boxes.row(i) = tracks[i].tlbr();
    }

    // Calculate IoU matrix
    Eigen::MatrixXd iou = box_iou_batch(track_boxes, tlbr_boxes);

    // Match detections with tracks based on IoU
    for (size_t i = 0; i < m; i++) {
        int idx_max = -1;
        double max_val = 0;
        for (size_t j = 0; j < n; j++) {
            if (iou(i, j) > max_val) {
                max_val = iou(i, j);
                idx_max = (int)j;
            }
        }

        // Assign track IDs based on highest IoU
        if (max_val > 0) {
            track_ids[idx_max] = tracks[i].get_track_id();
        }
    }

    return track_ids;
}
