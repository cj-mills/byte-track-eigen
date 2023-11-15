
#include "HungarianAlgorithmEigen.h"

/// <summary>
/// Default constructor for the HungarianAlgorithmEigen class.
/// </summary>
HungarianAlgorithmEigen::HungarianAlgorithmEigen() {}

/// <summary>
/// Destructor for the HungarianAlgorithmEigen class.
/// </summary>
HungarianAlgorithmEigen::~HungarianAlgorithmEigen() {}

/// <summary>
/// Finds the index of the first starred zero in a specified column.
/// </summary>
/// <param name="col">The column index to search within. Must be in the range of [0, number of columns in star_matrix).</param>
/// <returns>The row index of the starred zero, or NOT_FOUND_VALUE if no starred zero is found.</returns>
Eigen::Index HungarianAlgorithmEigen::find_star_in_column(int col)
{
    for (Eigen::Index i = 0; i < star_matrix.rows(); ++i) {
        if (star_matrix(i, col)) {
            return i; // Return the row index where the star was found.
        }
    }
    return NOT_FOUND_VALUE;
}

/// <summary>
/// Finds the index of the first primed zero in a specified row.
/// </summary>
/// <param name="row">The row index to search within.</param>
/// <returns>The column index of the primed zero, or NOT_FOUND_VALUE if no primed zero is found.</returns>
Eigen::Index HungarianAlgorithmEigen::find_prime_in_row(int row)
{
    for (Eigen::Index j = 0; j < prime_matrix.cols(); ++j) {
        if (prime_matrix(row, j)) {
            return j; // Return the column index where the prime was found.
        }
    }
    return NOT_FOUND_VALUE;
}

/// <summary>
/// Updates the star and prime matrices based on the given row and column.
/// It searches for a chain of alternating primed and starred zeros and
/// rebuilds the star matrix to include the new starred zero at the provided location.
/// </summary>
/// <param name="row">The row index of the primed zero to start with.</param>
/// <param name="col">The column index of the primed zero to start with.</param>
void HungarianAlgorithmEigen::update_star_and_prime_matrices(int row, int col)
{
    // Make a working copy of star_matrix
    new_star_matrix = star_matrix;

    // Star the current zero.
    new_star_matrix(row, col) = true;

    // Loop to update stars and primes based on the newly starred zero.
    while (true)
    {
        Eigen::Index star_row, prime_col;

        // If there are no starred zeros in the current column, break the loop.
        if (!star_matrix.col(col).any()) {
            break;
        }

        // Find the starred zero in the column and unstar it.
        star_row = find_star_in_column(col);
        new_star_matrix(star_row, col) = false;

        // If there are no primed zeros in the row of the starred zero, break the loop.
        if (!prime_matrix.row(star_row).any()) {
            break;
        }

        // Find the primed zero in the row and star it.
        prime_col = find_prime_in_row(star_row);
        new_star_matrix(star_row, prime_col) = true;

        // Move to the column of the newly starred zero.
        col = prime_col;
    }

    // Apply the changes from the working copy to the actual star_matrix.
    star_matrix = new_star_matrix;
    // Clear the prime_matrix and covered rows for the next steps.
    prime_matrix.setConstant(false);
    covered_rows.setConstant(false);
}

/// <summary>
/// Reduces the distance matrix by the smallest uncovered value and adjusts the
/// matrix to prepare for further steps of the Hungarian algorithm.
/// </summary>
void HungarianAlgorithmEigen::reduce_matrix_by_minimum_value()
{
    // Determine the dimensions for operations.
    int num_rows = covered_rows.size();
    int num_columns = covered_columns.size();

    // Create a masked array with high values for covered rows/columns.
    Eigen::ArrayXXd masked_array = dist_matrix.array() + DBL_MAX
        * (covered_rows.replicate(1, num_columns).cast<double>() + covered_columns.transpose().replicate(num_rows, 1).cast<double>());

    // Find the minimum value in the uncovered elements.
    double min_uncovered_value = masked_array.minCoeff();

    // Adjust the matrix values based on uncovered rows and columns.
    Eigen::ArrayXXd row_adjustments = covered_rows.cast<double>() * min_uncovered_value;
    Eigen::ArrayXXd col_adjustments = (1.0 - covered_columns.cast<double>()) * min_uncovered_value;
    dist_matrix += (row_adjustments.replicate(1, num_columns) - col_adjustments.transpose().replicate(num_rows, 1)).matrix();
}

/// <summary>
/// Covers columns that lack a starred zero and performs adjustments to the
/// distance matrix, primed and starred zeros until all columns are covered.
/// </summary>
void HungarianAlgorithmEigen::cover_columns_lacking_stars()
{
    // Retrieve the dimensions of the matrix.
    int num_rows = dist_matrix.rows();
    int num_columns = dist_matrix.cols();

    // Flag to check if uncovered zeros are found in the iteration.
    bool zeros_found = true;
    while (zeros_found)
    {
        zeros_found = false;
        for (int col = 0; col < num_columns; col++)
        {
            // Skip already covered columns.
            if (covered_columns(col)) continue;

            // Identify uncovered zeros in the current column.
            Eigen::Array<bool, Eigen::Dynamic, 1> uncovered_zeros_in_column = (dist_matrix.col(col).array().abs() < DBL_EPSILON) && !covered_rows;

            Eigen::Index row;
            // Check if there is an uncovered zero in the column.
            double max_in_uncovered_zeros = uncovered_zeros_in_column.cast<double>().maxCoeff(&row);

            // If an uncovered zero is found, prime it.
            if (max_in_uncovered_zeros == 1.0)
            {
                prime_matrix(row, col) = true;

                // Check for a star in the same row.
                Eigen::Index star_col;
                bool has_star = star_matrix.row(row).maxCoeff(&star_col);
                if (!has_star)
                {
                    // If no star is found, update the star and prime matrices, and covered columns.
                    update_star_and_prime_matrices(row, col);
                    covered_columns = (star_matrix.colwise().any()).transpose();
                    return;
                }
                else
                {
                    // If a star is found, cover the row and uncover the column where the star is found.
                    covered_rows(row) = true;
                    covered_columns(star_col) = false;
                    zeros_found = true; // Continue the while loop.
                    break;
                }
            }
        }
    }

    // If no more uncovered zeros are found, reduce the matrix and try again.
    reduce_matrix_by_minimum_value();
    cover_columns_lacking_stars();
}

/// <summary>
/// Executes the main steps of the Hungarian algorithm. It reduces the distance
/// matrix, stars zeros to form an initial feasible solution, and iteratively
/// improves the solution until an optimal assignment is found.
/// </summary>
void HungarianAlgorithmEigen::execute_hungarian_algorithm()
{
    // If there are fewer rows than columns, we operate on rows first.
    if (dist_matrix.rows() <= dist_matrix.cols())
    {
        for (int row = 0; row < dist_matrix.rows(); ++row)
        {
            // Subtract the minimum value in the row to create zeros.
            double min_value = dist_matrix.row(row).minCoeff();
            dist_matrix.row(row).array() -= min_value;

            // Identify zeros that are not covered by any line (row or column).
            Eigen::ArrayXd current_row = dist_matrix.row(row).array();
            Eigen::Array<bool, Eigen::Dynamic, 1> uncovered_zeros = (current_row.abs() < DBL_EPSILON) && !covered_columns;

            Eigen::Index col;
            // Star a zero if it is the only uncovered zero in its row.
            double max_in_uncovered_zeros = uncovered_zeros.cast<double>().maxCoeff(&col);
            if (max_in_uncovered_zeros == 1.0)
            {
                star_matrix(row, col) = true;
                covered_columns(col) = true; // Cover the column containing the starred zero.
            }
        }
    }
    else // If there are more rows than columns, we operate on columns first.
    {
        for (int col = 0; col < dist_matrix.cols(); ++col)
        {
            // Subtract the minimum value in the column to create zeros.
            double min_value = dist_matrix.col(col).minCoeff();
            dist_matrix.col(col).array() -= min_value;

            // Identify zeros that are not covered by any line.
            Eigen::ArrayXd current_column = dist_matrix.col(col).array();
            Eigen::Array<bool, Eigen::Dynamic, 1> uncovered_zeros = (current_column.abs() < DBL_EPSILON) && !covered_rows;

            Eigen::Index row;
            // Star a zero if it is the only uncovered zero in its column.
            double max_in_uncovered_zeros = uncovered_zeros.cast<double>().maxCoeff(&row);
            if (max_in_uncovered_zeros == 1.0)
            {
                star_matrix(row, col) = true;
                covered_columns(col) = true;
                covered_rows(row) = true; // Temporarily cover the row to avoid multiple stars in one row.
            }
        }

        // Uncover all rows for the next step.
        for (int row = 0; row < dist_matrix.rows(); ++row)
        {
            covered_rows(row) = false;
        }
    }

    // If not all columns are covered, move to the next step to cover them.
    if (covered_columns.count() != std::min(dist_matrix.rows(), dist_matrix.cols()))
    {
        cover_columns_lacking_stars();
    }
}

/// <summary>
/// Initializes helper arrays required for the Hungarian algorithm, setting the
/// appropriate sizes and default values.
/// </summary>
/// <param name="num_rows">The number of rows in the distance matrix.</param>
/// <param name="num_columns">The number of columns in the distance matrix.</param>
void HungarianAlgorithmEigen::init_helper_arrays(int num_rows, int num_columns) {
	covered_columns = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(num_columns, false);
	covered_rows = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(num_rows, false);
	star_matrix = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(num_rows, num_columns, false);
	new_star_matrix = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(num_rows, num_columns, false);
	prime_matrix = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(num_rows, num_columns, false);
}

/// <summary>
/// Constructs the assignment vector from the star matrix, providing the optimal
/// assignment of rows to columns.
/// </summary>
/// <param name="assignment">Reference to the vector where the result of the assignment will be stored.</param>
void HungarianAlgorithmEigen::construct_assignment_vector(Eigen::VectorXi& assignment)
{
    // Iterate over each row to determine the assignment for that row.
    for (int row = 0; row < star_matrix.rows(); ++row)
    {
        Eigen::Index col;
        // Check if there is a starred zero in the current row.
        bool has_star = star_matrix.row(row).maxCoeff(&col);
        if (has_star)
        {
            // If there is a starred zero, assign the corresponding column to this row.
            assignment[row] = col;
        }
        else
        {
            // If there is no starred zero, indicate that the row is not assigned.
            assignment[row] = DEFAULT_ASSIGNMENT_VALUE;
        }
    }
}


/// <summary>
/// Calculates the total cost of the assignment stored in the provided assignment vector.
/// </summary>
/// <param name="assignment">The vector containing the assignment for which to calculate the total cost.</param>
/// <returns>The total cost of the assignment.</returns>
double HungarianAlgorithmEigen::calculate_total_cost(Eigen::VectorXi& assignment)
{
    double total_cost = 0.0; // Initialize the total cost to zero.

    // Iterate over each assignment to calculate the total cost.
    for (int row = 0; row < dist_matrix.rows(); ++row)
    {
        // Check if the current row has a valid assignment.
        if (assignment(row) >= 0)
        {
            // Add the cost of the assigned column in the current row to the total cost.
            total_cost += dist_matrix(row, assignment(row));
        }
        // Note: If the assignment is not valid (indicated by DEFAULT_ASSIGNMENT_VALUE),
        // it is not included in the total cost calculation.
    }

    return total_cost; // Return the calculated total cost.
}

/// <summary>
/// Public interface to solve the assignment problem using the Hungarian algorithm.
/// This method sets up the problem using the given distance matrix and solves it,
/// returning the minimum total cost and filling the assignment vector with the
/// optimal assignment of rows to columns.
/// </summary>
/// <param name="dist_matrix">The distance matrix representing the cost of assigning rows to columns.</param>
/// <param name="assignment">Reference to the vector where the result of the assignment will be stored.</param>
/// <returns>The minimum total cost of the assignment.</returns>
double HungarianAlgorithmEigen::solve_assignment_problem(Eigen::MatrixXd& dist_matrix, Eigen::VectorXi& assignment)
{
    // Ensure that the distance matrix contains only non-negative values.
    if (dist_matrix.array().minCoeff() < 0)
    {
        std::cerr << "All matrix elements have to be non-negative." << std::endl;
    }

    // Copy the input distance matrix into the local member variable for manipulation.
    this->dist_matrix = dist_matrix;

    // Initialize helper arrays used in the algorithm, such as the star and prime matrices.
    init_helper_arrays(dist_matrix.rows(), dist_matrix.cols());

    // Execute the main steps of the Hungarian algorithm to find the optimal assignment.
    execute_hungarian_algorithm();

    // Set all elements of the assignment vector to a default value to indicate no assignment.
    assignment.setConstant(DEFAULT_ASSIGNMENT_VALUE);

    // Construct the assignment vector from the star matrix indicating the optimal assignment.
    construct_assignment_vector(assignment);

    // Calculate and return the total cost associated with the optimal assignment.
    return calculate_total_cost(assignment);
}