/**
 * @class HungarianAlgorithmEigen
 *
 * @brief Implements the Hungarian Algorithm using the Eigen library for solving the assignment problem.
 *
 * This class is designed to solve the assignment problem, which is to find the minimum cost way of assigning tasks to agents.
 * It makes use of the Eigen library for matrix operations, providing an efficient implementation suitable for large scale problems.
 *
 * Usage:
 * Eigen::MatrixXd cost_matrix = ...; // Define the cost matrix
 * Eigen::VectorXi assignment;
 * HungarianAlgorithmEigen solver;
 * double cost = solver.SolveAssignmentProblem(cost_matrix, assignment);
 *
 * Note:
 * The algorithm assumes that the input cost matrix contains only non-negative values. It is not thread-safe and is designed for single-threaded environments.
 */

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <cfloat>

// The HungarianAlgorithmEigen class encapsulates the Hungarian Algorithm
// for solving the assignment problem, which finds the minimum cost matching
// between elements of two sets. It uses the Eigen library to handle matrix operations.
class HungarianAlgorithmEigen
{
public:
    // Constructor and destructor
    HungarianAlgorithmEigen();
    ~HungarianAlgorithmEigen();

    // Solves the assignment problem given a distance matrix and returns the total cost.
    // The assignment of rows to columns is returned in the 'Assignment' vector.
    double solve_assignment_problem(Eigen::MatrixXd& dist_matrix, Eigen::VectorXi& assignment);

private:
    // Constants used as special values indicating not found or default assignments.
    const int NOT_FOUND_VALUE = -1;
    const int DEFAULT_ASSIGNMENT_VALUE = -1;

    // The distance matrix representing the cost of assignment between rows and columns.
    Eigen::MatrixXd dist_matrix;

    // Matrices and vectors used in the Hungarian algorithm
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> star_matrix, new_star_matrix, prime_matrix;
    Eigen::Array<bool, Eigen::Dynamic, 1> covered_columns, covered_rows;

    // Finds a starred zero in a column, if any.
    Eigen::Index find_star_in_column(int col);

    // Finds a primed zero in a row, if any.
    Eigen::Index find_prime_in_row(int row);

    // Updates the star and prime matrices given a row and column.
    void update_star_and_prime_matrices(int row, int col);

    // Reduces the matrix by subtracting the minimum value from each uncovered row
    // and adding it to each covered column, thus creating at least one new zero.
    void reduce_matrix_by_minimum_value();

    // Covers columns without a starred zero and potentially modifies star/prime matrices.
    void cover_columns_lacking_stars();

    // Executes the steps of the Hungarian algorithm.
    void execute_hungarian_algorithm();

    // Initializes helper arrays used by the algorithm.
    void init_helper_arrays(int num_rows, int num_columns);

    // Constructs the assignment vector from the star matrix.
    void construct_assignment_vector(Eigen::VectorXi& assignment);

    // Calculates the total cost of the assignment.
    double calculate_total_cost(Eigen::VectorXi& assignment);
};
