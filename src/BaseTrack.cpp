#include "BaseTrack.h"

// Initialize the static member variable _count to 0.
int BaseTrack::_count = 0;

// Default constructor: Initializes a new track with default values.
BaseTrack::BaseTrack() :
    track_id(0), 
    is_activated(false), 
    state(TrackState::New),
    curr_feature(0), 
    score(0), 
    start_frame(0), 
    frame_id(0),
    time_since_update(0),
    location(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()) 
{}

// Constructor with score: Initializes a new track with a specified score.
BaseTrack::BaseTrack(float score) :
    track_id(0),
    is_activated(false),
    state(TrackState::New),
    curr_feature(0),
    score(score),
    start_frame(0),
    frame_id(0),
    time_since_update(0),
    location(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())
{}

// end_frame: Returns the current frame ID of the track.
int BaseTrack::end_frame() const {
    return this->frame_id;
}

// next_id: Increments and returns the next unique track ID.
int BaseTrack::next_id() {
    return ++_count;
}

// activate: Placeholder for activating the track. Throws a runtime error if not implemented.
void BaseTrack::activate() {
    throw std::runtime_error("NotImplementedError");
}

// predict: Placeholder for predicting the next state of the track. Throws a runtime error if not implemented.
void BaseTrack::predict() {
    throw std::runtime_error("NotImplementedError");
}

// update: Placeholder for updating the track. Throws a runtime error if not implemented.
void BaseTrack::update() {
    throw std::runtime_error("NotImplementedError");
}

// mark_lost: Marks the track as lost.
void BaseTrack::mark_lost() {
    this->state = TrackState::Lost;
}

// mark_removed: Marks the track as removed.
void BaseTrack::mark_removed() {
    this->state = TrackState::Removed;
}

// resetCount: Resets the static track ID counter to 0.
void BaseTrack::reset_count() {
    _count = 0;
}

// Getter functions below provide safe access to the track's properties.

bool BaseTrack::get_is_activated() const {
    return this->is_activated;
}

TrackState BaseTrack::get_state() const {
    return this->state;
}

float BaseTrack::get_score() const {
    return this->score;
}

int BaseTrack::get_start_frame() const {
    return this->start_frame;
}

int BaseTrack::get_frame_id() const {
    return this->frame_id;
}

int BaseTrack::get_track_id() const {
    return this->track_id;
}