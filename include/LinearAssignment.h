#pragma once

#include <Eigen/Dense>
#include <vector>
#include <set>
#include <iostream>
#include <algorithm>
#include <tuple>

#include "HungarianAlgorithmEigen.h"

/**
 * @brief The LinearAssignment class encapsulates algorithms for solving the linear assignment problem.
 *
 * This class utilizes the Hungarian Algorithm for optimal assignment in a cost matrix. It's designed
 * to be used in contexts where assignments between different sets (like tasks to workers, or
 * observations to tracks) need to be made based on a cost matrix that quantifies the cost of each
 * possible assignment.
 *
 * Usage of the Eigen library enhances efficiency and allows for easy manipulation of matrices,
 * which are fundamental in the operations of this class.
 */
class LinearAssignment
{
private:
    HungarianAlgorithmEigen hungarian; // An instance of HungarianAlgorithmEigen to perform the actual assignment computations.

public:
    /**
     * @brief Generates matches based on a cost matrix and a set of indices, considering a threshold.
     *
     * @param cost_matrix The matrix representing the cost of assigning each pair of elements.
     * @param indices The matrix of indices representing potential matches.
     * @param thresh A threshold value to determine acceptable assignments. Assignments with a cost above this threshold won't be considered.
     * @return A tuple containing:
     *         1. A vector of pairs representing the selected assignments.
     *         2. A set of unassigned row indices.
     *         3. A set of unassigned column indices.
     */
    std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>> indices_to_matches(
        const Eigen::MatrixXd& cost_matrix,
        const Eigen::MatrixXi& indices,
        double thresh);

    /**
     * @brief Solves the linear assignment problem for a given cost matrix and threshold.
     *
     * This method is a straightforward approach to solve the linear assignment problem when all
     * elements of the cost matrix are considered for assignment.
     *
     * @param cost_matrix The matrix representing the cost of assigning each pair of elements.
     * @param thresh A threshold value to determine acceptable assignments. Assignments with a cost above this threshold won't be considered.
     * @return A tuple containing:
     *         1. A vector of pairs representing the selected assignments.
     *         2. A set of unassigned row indices.
     *         3. A set of unassigned column indices.
     */
    std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>> linear_assignment(
        const Eigen::MatrixXd& cost_matrix,
        double thresh
    );
};
