#include "BYTETracker.h"

/**
 * @brief Constructor for the BYTETracker class.
 *
 * Initializes the BYTETracker with specific tracking thresholds and parameters.
 * This setup is crucial for the tracking algorithm to adapt to different frame rates
 * and tracking conditions.
 *
 * @param track_thresh The threshold for considering a detection as a valid track.
 *        Detections with a score higher than this threshold will be considered for tracking.
 * @param track_buffer The size of the buffer to store the history of tracks.
 *        This parameter is used to calculate the maximum time a track can be lost.
 * @param match_thresh The threshold used for matching detections to existing tracks.
 *        A higher value requires a closer match between detection and track.
 * @param frame_rate The frame rate of the video being processed.
 *        It is used to adjust the buffer size relative to the BASE_FRAME_RATE.
 */
BYTETracker::BYTETracker(float track_thresh, int track_buffer, float match_thresh, int frame_rate) :
	track_thresh(track_thresh),
	match_thresh(match_thresh),
	frame_id(0),
	det_thresh(track_thresh + MIN_KEEP_THRESH),
	buffer_size(static_cast<int>(frame_rate / BASE_FRAME_RATE * track_buffer)),
	max_time_lost(buffer_size),
	kalman_filter(KalmanFilter())
{
	// Initialize tracking lists
	tracked_tracks.clear();
	lost_tracks.clear();
	removed_tracks.clear();

	// Reset the BaseTrack counter to ensure track IDs are unique per instance
	BaseTrack::reset_count();
}

/**
 * @brief Extracts Kalman bounding box tracks from detections.
 *
 * This method processes detection data and converts it into a vector of KalmanBBoxTrack objects.
 * Each detection is represented by a bounding box and associated confidence score.
 *
 * @param dets A matrix where each row represents a detected bounding box in the format (top-left and width-height coordinates).
 * @param scores_keep A vector of scores corresponding to the detections, indicating the confidence level of each detection.
 * @return std::vector<KalmanBBoxTrack> A vector of KalmanBBoxTrack objects representing the processed detections.
 */
std::vector<KalmanBBoxTrack> BYTETracker::extract_kalman_bbox_tracks(const Eigen::MatrixXf dets, const Eigen::VectorXf scores_keep) {

	std::vector<KalmanBBoxTrack> result;

	// Iterate through each detection and create a KalmanBBoxTrack object
	if (dets.rows() > 0) {
		for (int i = 0; i < dets.rows(); ++i) {
			Eigen::Vector4f tlwh = dets.row(i);
			// Create a KalmanBBoxTrack object with the converted bounding box and corresponding score
			result.push_back(KalmanBBoxTrack(std::vector<float>{ tlwh[0], tlwh[1], tlwh[2], tlwh[3]}, scores_keep[i]));
		}
	}

	return result;
}


/**
 * @brief Selects specific rows from a matrix based on given indices.
 *
 * This method is used to extract a subset of rows from a matrix, which is particularly
 * useful for processing detection data where only certain detections (rows) need to
 * be considered based on their scores or other criteria.
 *
 * @param matrix The input matrix from which rows will be selected.
 *        Each row in the matrix represents a distinct data point or detection.
 * @param indices A vector of indices indicating which rows should be selected from the matrix.
 *        The indices are zero-based and correspond to row numbers in the matrix.
 * @return Eigen::MatrixXf A new matrix containing only the rows specified by the indices.
 */
Eigen::MatrixXf BYTETracker::select_matrix_rows_by_indices(const Eigen::MatrixXf matrix, const std::vector<int> indices) {
	// Create a new matrix to hold the selected rows
	Eigen::MatrixXf result(indices.size(), matrix.cols());

	// Iterate over the provided indices and copy the corresponding rows to the result matrix
	for (int i = 0; i < indices.size(); ++i) {
		result.row(i) = matrix.row(indices[i]);
	}

	return result;
}

/**
 * @brief Filters and partitions detections based on confidence scores.
 *
 * This method processes the detection results, separating them into two groups based on their confidence scores.
 * One group contains detections with high confidence scores (above track_thresh),
 * and the other group contains detections with lower confidence scores (between MIN_KEEP_THRESH and track_thresh).
 * This separation is essential for handling detections differently based on their likelihood of being accurate.
 *
 * @param output_results A matrix containing the detection results. Each row represents a detection,
 *        typically including bounding box coordinates and a confidence score.
 * @return std::pair<std::vector<KalmanBBoxTrack>, std::vector<KalmanBBoxTrack>> A pair of vectors of KalmanBBoxTrack objects.
 *         The first vector contains high-confidence detections, and the second vector contains lower-confidence detections.
 */
std::pair<std::vector<KalmanBBoxTrack>, std::vector<KalmanBBoxTrack>> BYTETracker::filter_and_partition_detections(const Eigen::MatrixXf& output_results) {
	Eigen::VectorXf scores;
	Eigen::MatrixXf bboxes;

	// Extract scores and bounding boxes from output results
	// Assumes output_results contains bounding box coordinates followed by one score column
	scores = output_results.col(4);

	// Extract bounding box coordinates
	bboxes = output_results.leftCols(4);

	// Vectors to hold indices for high and low confidence detections
	std::vector<int> indices_high_thresh, indices_low_thresh;

	// Partition detections based on their scores
	for (int i = 0; i < scores.size(); ++i) {
		if (scores(i) > this->track_thresh) {
			indices_high_thresh.push_back(i);
		}
		else if (MIN_KEEP_THRESH < scores(i) && scores(i) < this->track_thresh) {
			indices_low_thresh.push_back(i);
		}
	}

	// Extract high and low confidence detections as KalmanBBoxTrack objects
	std::vector<KalmanBBoxTrack> detections = extract_kalman_bbox_tracks(select_matrix_rows_by_indices(bboxes, indices_high_thresh), select_matrix_rows_by_indices(scores, indices_high_thresh));
	std::vector<KalmanBBoxTrack> detections_second = extract_kalman_bbox_tracks(select_matrix_rows_by_indices(bboxes, indices_low_thresh), select_matrix_rows_by_indices(scores, indices_low_thresh));

	return { detections, detections_second };
}

/**
 * @brief Partition tracks into active and inactive based on their activation status.
 *
 * This method categorizes the currently tracked objects into two groups: active and inactive.
 * Active tracks are those that have been successfully associated with a detection in the
 * current frame or recent frames, indicating they are still visible. Inactive tracks are those
 * that have not been matched with a detection recently, suggesting they may be occluded or lost.
 *
 * @return A pair of vectors containing shared pointers to KalmanBBoxTrack objects.
 *         The first vector contains inactive tracks, and the second contains active tracks.
 */
std::pair<std::vector<std::shared_ptr<KalmanBBoxTrack>>, std::vector<std::shared_ptr<KalmanBBoxTrack>>> BYTETracker::partition_tracks_by_activation() {
	std::vector<std::shared_ptr<KalmanBBoxTrack>> inactive_tracked_tracks;
	std::vector<std::shared_ptr<KalmanBBoxTrack>> active_tracked_tracks;

	// Iterate through all tracked tracks and partition them based on their activation status
	for (auto& track : this->tracked_tracks) {
		if (track->get_is_activated()) {
			active_tracked_tracks.push_back(track); // Add to active tracks if the track is activated
		}
		else {
			inactive_tracked_tracks.push_back(track); // Otherwise, consider it as inactive
		}
	}

	return { inactive_tracked_tracks, active_tracked_tracks };
}

/**
 * @brief Assigns tracks to detections based on a specified threshold.
 *
 * This method performs the critical task of associating existing tracks with new detections.
 * It uses the Intersection over Union (IoU) metric to measure the similarity between
 * each track and detection, and then applies the Hungarian algorithm (via the LinearAssignment class)
 * to find the optimal assignment between tracks and detections.
 *
 * @param tracks The vector of shared pointers to KalmanBBoxTrack objects representing the current tracks.
 * @param detections The vector of KalmanBBoxTrack objects representing the new detections for the current frame.
 * @param thresh The threshold for the IoU similarity measure. Pairs with an IoU below this threshold will not be assigned.
 * @return A tuple containing three elements:
 *         - A vector of pairs, where each pair contains the index of a track and the index of a matched detection.
 *         - A set of integers representing the indices of tracks that couldn't be paired with any detection.
 *         - A set of integers representing the indices of detections that couldn't be paired with any track.
 */
std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>> BYTETracker::assign_tracks_to_detections(
	const std::vector<std::shared_ptr<KalmanBBoxTrack>> tracks,
	const std::vector<KalmanBBoxTrack> detections,
	double thresh
) {
	// Convert shared pointers to instances for distance computation
	std::vector<KalmanBBoxTrack> track_instances;
	track_instances.reserve(tracks.size());
	for (const auto& ptr : tracks) {
		track_instances.push_back(*ptr);
	}

	// Compute the IoU distance matrix between tracks and detections
	Eigen::MatrixXd distances = iou_distance(track_instances, detections);

	// Perform linear assignment to find the best match between tracks and detections
	return this->linear_assignment.linear_assignment(distances, thresh);
}

/**
 * @brief Updates tracks with the latest detection information.
 *
 * This method is responsible for updating the states of the tracks based on the new detections.
 * It involves matching detected objects to existing tracks and updating the track state accordingly.
 * The method also handles reactivating tracks that were previously lost and categorizing them as either
 * reacquired or newly activated.
 *
 * @param tracks A reference to a vector of shared pointers to KalmanBBoxTrack objects representing the current tracks.
 * @param detections A vector of KalmanBBoxTrack objects representing the new detections in the current frame.
 * @param track_detection_pair_indices A vector of pairs, where each pair contains the index of a track and
 *        the index of a detection that have been matched together.
 * @param reacquired_tracked_tracks A reference to a vector where reactivated tracks will be stored.
 * @param activated_tracks A reference to a vector where newly activated tracks will be stored.
 */
void BYTETracker::update_tracks_from_detections(
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& tracks,
	const std::vector<KalmanBBoxTrack> detections,
	const std::vector<std::pair<int, int>> track_detection_pair_indices,
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& reacquired_tracked_tracks,
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& activated_tracks
) {

	for (const auto match : track_detection_pair_indices) {
		if (tracks[match.first]->get_state() == TrackState::Tracked) {
			// Update existing tracked track with the new detection
			tracks[match.first]->update(detections[match.second], this->frame_id);
			activated_tracks.push_back(tracks[match.first]);
		}
		else {
			// Reactivate a track that was previously lost
			tracks[match.first]->re_activate(detections[match.second], this->frame_id, false);
			reacquired_tracked_tracks.push_back(tracks[match.first]);
		}
	}
}

/**
 * @brief Extracts active tracks from a given set of tracks.
 *
 * This method filters through a collection of tracks and extracts those that are actively
 * being tracked (i.e., their state is marked as 'Tracked'). It uses a set of indices to identify
 * unpaired tracks, ensuring that only relevant and currently tracked objects are considered.
 * This function is essential in the tracking process, where it's crucial to distinguish between
 * actively tracked, lost, and new objects.
 *
 * @param tracks A vector of shared pointers to KalmanBBoxTrack objects, representing all currently known tracks.
 * @param unpaired_track_indices A set of integers representing the indices of tracks that have not been paired
 *        with a detection in the current frame. This helps in identifying which tracks are still active.
 * @return std::vector<std::shared_ptr<KalmanBBoxTrack>> A vector of shared pointers to KalmanBBoxTrack objects
 *         that are actively being tracked.
 */
std::vector<std::shared_ptr<KalmanBBoxTrack>> BYTETracker::extract_active_tracks(
	const std::vector<std::shared_ptr<KalmanBBoxTrack>>& tracks,
	std::set<int> unpaired_track_indices
) {
	std::vector<std::shared_ptr<KalmanBBoxTrack>> currently_tracked_tracks;
	for (int i : unpaired_track_indices) {
		if (i < tracks.size() && tracks[i]->get_state() == TrackState::Tracked) {
			currently_tracked_tracks.push_back(tracks[i]);
		}
	}
	return currently_tracked_tracks;
}

/**
 * @brief Flags unpaired tracks as lost.
 *
 * This method takes a list of currently tracked tracks and a set of unpaired track indices,
 * marking those unpaired tracks as 'lost'. It helps in updating the state of tracks that
 * are no longer detected in the current frame. This is an essential step in the tracking
 * process as it assists in handling temporary occlusions or missed detections.
 *
 * @param currently_tracked_tracks A vector of shared pointers to KalmanBBoxTrack objects representing
 *        the currently active tracks. These are the tracks that are being updated in the current frame.
 * @param lost_tracks A vector to which lost tracks will be added. These tracks were not matched with
 *        any current detection and are thus considered lost.
 * @param unpaired_track_indices A set of indices pointing to tracks in currently_tracked_tracks
 *        that have not been paired with any detection in the current frame.
 */
void BYTETracker::flag_unpaired_tracks_as_lost(
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& currently_tracked_tracks,
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& lost_tracks,
	std::set<int> unpaired_track_indices
) {
	for (int i : unpaired_track_indices) {
		// Check if the index is within bounds and the track state is not already lost
		if (i < currently_tracked_tracks.size() && currently_tracked_tracks[i]->get_state() != TrackState::Lost) {
			// Mark the track as lost and add it to the lost_tracks vector
			currently_tracked_tracks[i]->mark_lost();
			lost_tracks.push_back(currently_tracked_tracks[i]);
		}
	}
}

/**
 * @brief Prunes and merges tracked tracks.
 *
 * This method updates the state of the tracked tracks by pruning tracks that are no longer in
 * the 'Tracked' state and merging the list of tracks with newly activated and reacquired tracks.
 * It ensures that the tracked_tracks list always contains the most up-to-date and relevant tracks.
 *
 * @param reacquired_tracked_tracks A vector of tracks that have been reacquired after being lost.
 *        These tracks are reintegrated into the main tracking list.
 * @param activated_tracks A vector of newly activated tracks that need to be added to the main tracking list.
 */
void BYTETracker::prune_and_merge_tracked_tracks(
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& reacquired_tracked_tracks,
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& activated_tracks
) {
	// Update tracked_tracks to only contain tracks that are in the Tracked state
	std::vector<std::shared_ptr<KalmanBBoxTrack>> filtered_tracked_tracks;
	for (std::shared_ptr<KalmanBBoxTrack> track : this->tracked_tracks) {
		if (track->get_state() == TrackState::Tracked) {
			filtered_tracked_tracks.push_back(track);
		}
	}
	this->tracked_tracks = filtered_tracked_tracks;

	// Update tracked_tracks by merging with activated and reacquired tracks
	this->tracked_tracks = join_tracks(this->tracked_tracks, activated_tracks);
	this->tracked_tracks = join_tracks(this->tracked_tracks, reacquired_tracked_tracks);
}

/**
 * @brief Handles the updating of lost and removed track lists.
 *
 * This method updates the internal lists of lost and removed tracks based on the current frame.
 * Tracks that have been lost for a duration longer than the maximum allowable time (max_time_lost)
 * are marked as removed. The method also ensures that the lists of lost and removed tracks are
 * properly maintained and updated, considering the current state of tracked and lost tracks.
 *
 * @param removed_tracks A reference to a vector of shared pointers to KalmanBBoxTrack, representing
 *        the tracks that are currently marked as removed.
 * @param lost_tracks A reference to a vector of shared pointers to KalmanBBoxTrack, representing
 *        the tracks that are currently marked as lost.
 */
void BYTETracker::handle_lost_and_removed_tracks(
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& removed_tracks,
	std::vector<std::shared_ptr<KalmanBBoxTrack>>& lost_tracks
) {
	// Iterate over lost tracks and mark them as removed if they have been lost for too long
	for (std::shared_ptr<KalmanBBoxTrack> track : this->lost_tracks) {
		if (this->frame_id - track->end_frame() > this->max_time_lost) {
			track->mark_removed();
			removed_tracks.push_back(track);
		}
	}

	// Update the lost_tracks list by removing tracks that are currently being tracked
	// or have been marked as removed
	this->lost_tracks = sub_tracks(this->lost_tracks, this->tracked_tracks);
	this->lost_tracks.insert(this->lost_tracks.end(), lost_tracks.begin(), lost_tracks.end());
	this->lost_tracks = sub_tracks(this->lost_tracks, this->removed_tracks);

	// Clean up removed tracks
	this->removed_tracks.clear();
}

/**
 * @brief Processes detections for a single frame and updates track states.
 *
 * This method is the core of the BYTETracker class, where detections in each frame
 * are processed to update the state of the tracks. It involves several key steps:
 * - Partitioning detections based on confidence thresholds.
 * - Predicting the state of existing tracks.
 * - Matching detections to existing tracks and updating their states.
 * - Handling unpaired tracks and detections.
 * - Managing lost and new tracks.
 *
 * @param output_results A matrix containing detection data for the current frame.
 *        Each row represents a detection and includes bounding box coordinates 
 * 		  in the format (top-left and width-height coordinates) and a detection score.
 * @return std::vector<KalmanBBoxTrack> A vector of KalmanBBoxTrack objects representing
 *         the updated state of each track after processing the current frame.
 */
std::vector<KalmanBBoxTrack> BYTETracker::process_frame_detections(const Eigen::MatrixXf& output_results) {
	// Increment the frame counter
	this->frame_id += 1;

	// Initialize containers for various track states
	std::vector<std::shared_ptr<KalmanBBoxTrack>> reacquired_tracked_tracks, activated_tracks, lost_tracks, removed_tracks;

	// Filter and partition detections based on confidence thresholds
	auto [high_confidence_detections, lower_confidence_detections] = filter_and_partition_detections(output_results);

	// Partition existing tracks into active and inactive ones
	auto [inactive_tracked_tracks, active_tracked_tracks] = partition_tracks_by_activation();

	// Prepare track pool for matching and update state prediction for each track
	std::vector<std::shared_ptr<KalmanBBoxTrack>> track_pool = join_tracks(active_tracked_tracks, this->lost_tracks);
	KalmanBBoxTrack::multi_predict(track_pool);

	// Match tracks to high confidence detections and update their states
	auto [track_detection_pair_indices, unpaired_track_indices, unpaired_detection_indices] = assign_tracks_to_detections(track_pool, high_confidence_detections, this->match_thresh);
	update_tracks_from_detections(track_pool, high_confidence_detections, track_detection_pair_indices, reacquired_tracked_tracks, activated_tracks);

	// Extract currently tracked tracks from the pool
	auto currently_tracked_tracks = extract_active_tracks(track_pool, unpaired_track_indices);

	// Match currently tracked tracks to lower confidence detections and update states
	std::tie(track_detection_pair_indices, unpaired_track_indices, std::ignore) = assign_tracks_to_detections(currently_tracked_tracks, lower_confidence_detections, LOWER_CONFIDENCE_MATCHING_THRESHOLD);
	update_tracks_from_detections(currently_tracked_tracks, lower_confidence_detections, track_detection_pair_indices, reacquired_tracked_tracks, activated_tracks);

	// Flag unpaired tracks as lost
	flag_unpaired_tracks_as_lost(currently_tracked_tracks, lost_tracks, unpaired_track_indices);

	// Update unconfirmed tracks
	std::vector<KalmanBBoxTrack> filtered_detections;
	for (int i : unpaired_detection_indices) {
		filtered_detections.push_back(high_confidence_detections[i]);
	}
	high_confidence_detections = filtered_detections;

	// Match inactive tracks to high confidence detections for reactivation
	std::tie(track_detection_pair_indices, unpaired_track_indices, unpaired_detection_indices) = assign_tracks_to_detections(inactive_tracked_tracks, high_confidence_detections, ACTIVATION_MATCHING_THRESHOLD);
	for (auto [track_idx, det_idx] : track_detection_pair_indices) {
		inactive_tracked_tracks[track_idx]->update(high_confidence_detections[det_idx], this->frame_id);
		activated_tracks.push_back(inactive_tracked_tracks[track_idx]);
	}

	// Handle tracks that remain inactive
	for (int i : unpaired_track_indices) {
		inactive_tracked_tracks[i]->mark_removed();
		removed_tracks.push_back(inactive_tracked_tracks[i]);
	}

	// Handle new tracks from unpaired detections
	for (int i : unpaired_detection_indices) {
		if (high_confidence_detections[i].get_score() >= this->det_thresh) {
			high_confidence_detections[i].activate(this->kalman_filter, this->frame_id);
			activated_tracks.push_back(std::make_shared<KalmanBBoxTrack>(high_confidence_detections[i]));
		}
	}

	// Merge and prune tracked tracks
	prune_and_merge_tracked_tracks(reacquired_tracked_tracks, activated_tracks);

	// Handle lost tracks and remove outdated ones
	handle_lost_and_removed_tracks(removed_tracks, lost_tracks);

	// Consolidate and clean up track lists
	this->removed_tracks.insert(this->removed_tracks.end(), removed_tracks.begin(), removed_tracks.end());

	// Remove duplicate tracks
	std::tie(this->tracked_tracks, this->lost_tracks) = remove_duplicate_tracks(this->tracked_tracks, this->lost_tracks);

	// Prepare the list of tracks to be returned
	std::vector<KalmanBBoxTrack> return_tracks;
	for (std::shared_ptr<KalmanBBoxTrack> track : this->tracked_tracks) {
		if (track->get_is_activated()) {
			return_tracks.push_back(*track);
		}
	}

	return return_tracks;
}
