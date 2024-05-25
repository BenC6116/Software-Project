import sys
import numpy as np
import pandas as pd
import mysymnmf

MAX_ITER = 300
EPSILON = 1e-4
GENERAL_ERROR = "An Error Has Occurred"

# This function takes a matrix and prints it in a formatted way.
# Each row is printed on a new line, and the columns of the matrix are separated by commas.
def print_matrix(matrix):
    my_string = ''
    for row in matrix:
        my_string += ','.join(map(lambda x: "{:.4f}".format(x), row)) + '\n'
    my_string = my_string.rstrip('\n')
    print(my_string)


# This function performs step 1.4 in the project instructions
def get_symnmf(n, k, dim, data_points):
    np.random.seed(0)
    normal_sim = mysymnmf.norm(n, dim, data_points)  # Get W (the normalized similarity matrix) from C
    norm_sim_mean = np.mean(normal_sim)
    upper_bound = 2 * np.sqrt(norm_sim_mean / k) # 2 * sqrt(m/k)

    # 1.4.1 - Initialize H
    decomp = np.random.uniform(0, upper_bound, size=(n,k))
    decomp = decomp.tolist()
    # Call the symnmf method
    return mysymnmf.symnmf(n, k, decomp, normal_sim)


# The main function of the program. It takes command-line arguments, validates them, performs
# the required algorithm, and then prints the result.
def main():
    program_terminator = False

    if len(sys.argv) != 4:
        print(GENERAL_ERROR)
        program_terminator = True

    # Get input from cmd
    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_name = sys.argv[3]

    file_df = pd.read_csv(file_name, header=None)  # Create a pandas data frame for file_name
    # Validate the number of clusters. It should be greater than 1 and less than the number of data points.
    if k <= 1 or k >= file_df.shape[0]:
        print(GENERAL_ERROR)
        program_terminator = True

    # If either the number of clusters or the maximum number of iterations is invalid, terminate the program.
    if program_terminator:
        quit()

    # From HW2 - Might be helpful
    dim = file_df.shape[1]
    data_points = file_df.values.tolist()
    n = file_df.shape[0]
    
    matrix = None
    try:
        if goal == 'sym':
            # Call the main C sym method
            matrix = mysymnmf.sym(n, dim, data_points)
        elif goal == 'ddg':
            # Call the main C ddg method
            matrix = mysymnmf.ddg(n, dim, data_points)
        elif goal == 'norm':
            # Call the main C norm method
            matrix = mysymnmf.norm(n, dim, data_points)
        elif goal == 'symnmf':
            matrix = get_symnmf(n,k,dim,data_points)
        else:
            raise Exception(GENERAL_ERROR)

        print_matrix(matrix)
    except RuntimeError:
        print(GENERAL_ERROR)
        quit()


# Run the main function if this script is run as the main module.
if __name__ == "__main__":
    main()