#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <stdexcept>
#include <limits>

// Enum class for defining the state of a track.
enum class TrackState {
    New,      // Indicates a newly created track.
    Tracked,  // Indicates a track that is currently being tracked.
    Lost,     // Indicates a track that has been lost.
    Removed   // Indicates a track that has been removed.
};

/**
 * @brief The BaseTrack class serves as the foundational class for object tracking.
 *        It defines the essential attributes and methods for a track, such as ID, state, score, features, and history.
 *        The class provides mechanisms to handle the state of a track, including creating, updating, and maintaining its lifecycle.
 *        Designed to be extended, it allows for specific tracking functionalities to be implemented in derived classes.
 */
class BaseTrack {
private:
    static int _count; // Static counter used to assign unique IDs to each track.

protected:
    int track_id; // Unique identifier for the track.
    bool is_activated; // Flag to indicate if the track is active.
    TrackState state; // Current state of the track.
    std::map<int, int> history; // History of track updates, typically storing frame ID and associated data.
    std::vector<int> features; // Features associated with the track.
    int curr_feature; // Current feature of the track.
    float score; // Score associated with the track, indicating the confidence of tracking.
    int start_frame; // Frame at which the track was started.
    int frame_id; // Current frame ID of the track.
    int time_since_update; // Time elapsed since the last update.
    std::pair<double, double> location; // Current location of the track.

public:
    // Default constructor.
    BaseTrack();

    // Constructor with score initialization.
    BaseTrack(float score);

    // Returns the end frame of the track.
    int end_frame() const;

    // Generates the next unique track ID.
    static int next_id();

    // Activates the track. This method is intended to be overridden in derived classes.
    virtual void activate();

    // Predicts the next state of the track. This method is intended to be overridden in derived classes.
    virtual void predict();

    // Updates the track with new data. This method is intended to be overridden in derived classes.
    virtual void update();

    // Marks the track as lost.
    void mark_lost();

    // Marks the track as removed.
    void mark_removed();

    // Resets the static track ID counter. Useful for unit testing or reinitializing the tracking system.
    static void reset_count();

    // Getter for is_activated.
    bool get_is_activated() const;

    // Getter for state.
    TrackState get_state() const;

    // Getter for score.
    float get_score() const;

    // Getter for start_frame.
    int get_start_frame() const;

    // Getter for frame_id.
    int get_frame_id() const;

    // Getter for track_id.
    int get_track_id() const;
};
