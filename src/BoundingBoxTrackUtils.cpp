#include "BoundingBoxTrackUtils.h"

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
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_b) {

    std::map<int, std::shared_ptr<KalmanBBoxTrack>> unique_tracks;

    // Populate the map with tracks from track_list_a
    for (auto& track : track_list_a) {
        unique_tracks[track->get_track_id()] = track;
    }

    // Insert or overwrite with tracks from track_list_b
    for (auto& track : track_list_b) {
        unique_tracks[track->get_track_id()] = track;
    }

    // Convert the unique tracks in the map to a vector
    std::vector<std::shared_ptr<KalmanBBoxTrack>> result;
    for (auto [key, value] : unique_tracks) {
        result.push_back(value);
    }

    return result;
}

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
    const std::vector<std::shared_ptr<KalmanBBoxTrack>>& track_list_b) {

    std::unordered_set<int> track_ids_b;

    // Populate set with track IDs from track_list_b for quick lookup
    for (const auto& track : track_list_b) {
        track_ids_b.insert(track->get_track_id());
    }

    // Collect tracks from track_list_a not found in track_list_b
    std::vector<std::shared_ptr<KalmanBBoxTrack>> result;
    for (const auto& track : track_list_a) {
        if (track_ids_b.find(track->get_track_id()) == track_ids_b.end()) {
            result.push_back(track);
        }
    }

    return result;
}

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
std::pair<std::vector<std::shared_ptr<KalmanBBoxTrack>>, std::vector<std::shared_ptr<KalmanBBoxTrack>>>
remove_duplicate_tracks(const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_a,
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_b) {

    // Creating instance vectors to store de-referenced shared pointers from the input lists.
    // This is necessary for the IOU distance calculation.
    std::vector<KalmanBBoxTrack> s_tracks_a_instances;
    s_tracks_a_instances.reserve(track_list_a.size());
    for (const auto& ptr : track_list_a) {
        s_tracks_a_instances.push_back(*ptr);
    }

    std::vector<KalmanBBoxTrack> s_tracks_b_instances;
    s_tracks_b_instances.reserve(track_list_b.size());
    for (const auto& ptr : track_list_b) {
        s_tracks_b_instances.push_back(*ptr);
    }

    // Calculating pairwise IOU distance between tracks in both lists.
    Eigen::MatrixXd pairwise_distance = iou_distance(s_tracks_a_instances, s_tracks_b_instances);

    // Sets to track indices of duplicates in both lists.
    std::unordered_set<int> duplicates_a, duplicates_b;

    // Loop through the matrix of pairwise distances to identify duplicates.
    for (int i = 0; i < pairwise_distance.rows(); ++i) {
        for (int j = 0; j < pairwise_distance.cols(); ++j) {
            // If IOU distance is below the threshold, consider the tracks as duplicates.
            if (pairwise_distance(i, j) < 0.15) {
                // Calculate track age to determine which track to keep.
                int time_a = track_list_a[i]->get_frame_id() - track_list_a[i]->get_start_frame();
                int time_b = track_list_b[j]->get_frame_id() - track_list_b[j]->get_start_frame();

                // Retain the older track and mark the newer one as a duplicate.
                if (time_a > time_b) {
                    duplicates_b.insert(j);
                }
                else {
                    duplicates_a.insert(i);
                }
            }
        }
    }

    // Constructing the result lists, excluding the identified duplicates.
    std::vector<std::shared_ptr<KalmanBBoxTrack>> result_a, result_b;
    for (int i = 0; i < track_list_a.size(); ++i) {
        if (duplicates_a.find(i) == duplicates_a.end()) {
            result_a.push_back(track_list_a[i]);
        }
    }
    for (int j = 0; j < track_list_b.size(); ++j) {
        if (duplicates_b.find(j) == duplicates_b.end()) {
            result_b.push_back(track_list_b[j]);
        }
    }

    return { result_a, result_b };
}

