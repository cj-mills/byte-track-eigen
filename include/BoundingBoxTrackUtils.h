#pragma once

#include <vector>
#include <map>
#include <unordered_set>
#include <Eigen/Dense>
#include "KalmanBBoxTrack.h"
#include "BoundingBoxIoUMatching.h"

/**
 * @brief Joins two lists of KalmanBBoxTrack pointers, ensuring unique track IDs.
 *
 * This function combines track lists A and B. If a track ID is present in both lists,
 * the track from list B is used in the final list. This ensures that each track ID is
 * unique in the resulting list.
 *
 * @param track_list_a First list of shared pointers to KalmanBBoxTrack objects.
 * @param track_list_b Second list of shared pointers to KalmanBBoxTrack objects.
 * @return std::vector<std::shared_ptr<KalmanBBoxTrack>> A list with unique tracks based on track IDs.
 */
std::vector<std::shared_ptr<KalmanBBoxTrack>> join_tracks(
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_a,
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_b);

/**
 * @brief Subtracts tracks in track_list_b from track_list_a based on track IDs.
 *
 * This function creates a new list of tracks from track_list_a excluding those
 * tracks whose IDs are found in track_list_b, effectively performing a set subtraction.
 *
 * @param track_list_a First list of shared pointers to KalmanBBoxTrack objects.
 * @param track_list_b Second list of shared pointers to KalmanBBoxTrack objects.
 * @return std::vector<std::shared_ptr<KalmanBBoxTrack>> A list of tracks present in track_list_a but not in track_list_b.
 */
std::vector<std::shared_ptr<KalmanBBoxTrack>> sub_tracks(
    const std::vector<std::shared_ptr<KalmanBBoxTrack>>& track_list_a,
    const std::vector<std::shared_ptr<KalmanBBoxTrack>>& track_list_b);

/**
 * @brief Removes duplicate tracks from two lists based on IOU distance and track age.
 *
 * This function identifies and removes duplicate tracks between two lists based on the
 * Intersection Over Union (IOU) distance. If two tracks have an IOU distance less than
 * a threshold (0.15), the older track is retained.
 *
 * @param track_list_a First list of shared pointers to KalmanBBoxTrack objects.
 * @param track_list_b Second list of shared pointers to KalmanBBoxTrack objects.
 * @return Pair of vectors each containing unique tracks after removing duplicates.
 */
std::pair<std::vector<std::shared_ptr<KalmanBBoxTrack>>, std::vector<std::shared_ptr<KalmanBBoxTrack>>> remove_duplicate_tracks(
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_a,
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_b);
