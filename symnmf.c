#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const char* GENERAL_ERROR = "An Error Has Occurred";

/*
 * This program parses an input data file containing point coordinates.
 * It computes three types of matrices based on the points: symmetric, diagonal degree,
 * and normalized Laplacian matrix. The matrix to compute is specified as a command-line argument.
 */

/* Utility function: Calculates the squared Euclidean distance between two vectors. */
double euc_dist_square(double *x, double *y, int d) {
    int i;
    double sum = 0;
    for (i = 0; i < d; i++) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sum;
}

/* Utility function: Modifies a diagonal matrix in place to have the square root of its inverse values. */
void get_diag_inv_square(double **diag_matrix, int n) {
    int i;
    for(i=0; i<n; i++) {
        diag_matrix[i][i] = 1/sqrt(diag_matrix[i][i]);
    }
}

/*
 * Utility function: Multiplies two matrices.
 * Assumes square matrices of the same dimension.
 */
void matrix_mult(int rowsA, int sharedDim, int colsB, 
                       double** matrixA, double** matrixB, double** resultMatrix) {
    int row, col, inner;
    for (row = 0; row < rowsA; row++) {
        for (col = 0; col < colsB; col++) {
            resultMatrix[row][col] = 0.0;
            for (inner = 0; inner < sharedDim; inner++) {
                resultMatrix[row][col] += matrixA[row][inner] * matrixB[inner][col];
            }
        }
    }
}

/*
 * Computes the symmetric matrix based on the given points.
 * The matrix entry (i, j) is the exponentiated negative half of the squared Euclidean distance between points i and j.
 */
void get_sym(double **points_array, int dim, int n, double **sym_matrix) {
    int i,j;
    for(i =0; i<n; i++) {
        for(j=0; j<n;j++) {
            if(i==j) {
                sym_matrix[i][j] = 0;
            } else {
                sym_matrix[i][j] = exp(-0.5*euc_dist_square(points_array[i], points_array[j],dim));
            }
        }
    }
}

/*
 * Computes the diagonal degree matrix based on the given points.
 * Each diagonal entry is the sum of the respective row in the symmetric matrix.
 */
void get_ddg(double **points_array, int m, int n, double **diag_matrix) {
    int i,j,k;
    double **sym_matrix = malloc(n*sizeof(double*));
    double vector_sum = 0;
    for(i=0; i<n;i++) {
        sym_matrix[i] = malloc(n*sizeof(double));

    }
    get_sym(points_array, m, n, sym_matrix);

    for(i =0; i<n; i++) {
        for(j=0; j<n;j++) {
            if(i!=j) {
                diag_matrix[i][j] = 0;
            } else {
                vector_sum=0;
                for(k=0;k<n;k++) {
                    vector_sum+=sym_matrix[i][k];
                }
                diag_matrix[i][j] = vector_sum;
            }
        }
    
    }

    for(i=0; i<n;i++) {
        free(sym_matrix[i]);
    }
    free(sym_matrix);
}

/*
 * Computes the normalized Laplacian matrix based on the given points.
 */
void get_norm(double **points_array, int m, int n, double **norm_matrix) {
    int i;
    double **ddg_matrix = malloc(n*sizeof(double*));
    double **sym_matrix = malloc(n*sizeof(double*));
    double **temp_matrix = malloc(n*sizeof(double*));
    for(i=0; i<n;i++) {
        ddg_matrix[i] = malloc(n*sizeof(double));
    }
    get_ddg(points_array, m, n, ddg_matrix);

    for(i=0; i<n;i++) {
        sym_matrix[i] = malloc(n*sizeof(double));
    }
    get_sym(points_array, m, n, sym_matrix);

    get_diag_inv_square(ddg_matrix,n);
    
    for(i=0; i<n;i++) {
        temp_matrix[i] = malloc(n*sizeof(double));
    }

    matrix_mult(n,n,n, ddg_matrix, sym_matrix, temp_matrix);
    matrix_mult(n,n,n, temp_matrix, ddg_matrix, norm_matrix);

    for(i=0; i<n;i++) {
        free(sym_matrix[i]);
    }
    free(sym_matrix);

    for(i=0; i<n;i++) {
        free(ddg_matrix[i]);
    }
    free(ddg_matrix);

    for(i=0; i<n;i++) {
        free(temp_matrix[i]);
    }
    free(temp_matrix);
}

/*
 * This function reads the first line of the input file to determine
 * the number of dimensions by counting the number of commas.
 */
int find_dimension(FILE* input_file) {
    int comma_count = 1;
    char ch;

    while ((ch = fgetc(input_file)) != EOF && ch != '\n') {
        if (ch == ',') {
            comma_count++;
        }
    }
    rewind(input_file);
    return comma_count;
}

/*
 * This function splits the given string by commas and converts
 * each substring to a double, storing the result in the provided array.
 */
void split_string_to_doubles(char* str, double* values, int dim) {
    int index;
    int i;
    char* current_double_pointer;
    char delim;

    delim = ',';

    index = 0;
    current_double_pointer = str;
    for (i = 0; i < dim; i++) {
        while (str[index] != delim && str[index] != '\0') {
            index++;
        }
        str[index] = '\0';
        values[i] = strtod(current_double_pointer, NULL);

        current_double_pointer = &str[index + 1];
        index++;
    }
}

/*
 * This function reads through the provided file and counts the number
 * of newline characters to determine the number of lines.
 */
int count_lines(FILE *file) {
    int count = 0;
    char ch1 = ' ';
    char ch2 = ' ';
    if (!file) {
        return -1; /*ERROR INDICATOR*/
    }
    while (ch1 != EOF) {
        ch2 = ch1;
        ch1 = fgetc(file);
        if (ch1 == '\n') {
            if(ch2 >= '0' || ch2 <= '9')
                count++;
        }
    }

    if(ch2 != '\n')
        count++;
    
    /*restore original file position*/
    rewind(file);
    return count;
}

/*
 * Reads the input file line by line, splits each line into a list of doubles,
 * and appends each list to a dynamically allocated array.
 */
double** get_data_points_array(FILE* input_file, int dim, int count_lines) {
    double **points_array = malloc(count_lines* sizeof(double*));
    double* values = NULL;
    char *values_buffer = malloc(dim*30);
    int i,j;

    memset(values_buffer,0,dim*30);
    for(i=0; i<count_lines; i++) {
        if(fgets(values_buffer, dim*30, input_file) == NULL){
            free(values_buffer);
            for (j = 0; j < i; j++) {
                free(points_array[j]);
            }
            free(points_array);
            return NULL;
        }
        values = malloc(dim * sizeof(double));
        split_string_to_doubles(values_buffer, values, dim);

        points_array[i] = values;
        memset(values_buffer,0,dim*30);
    }
    free(values_buffer);
    return points_array;
}

/*
 * Uses utility functions to determine the dimensionality of the data and the number of data points.
 * It then reads and parses the data points from the file.
 */
double **parse_data(FILE* input_file, int* dim_ptr, int* count_lines_ptr) {
    double **points_array = NULL;
    *count_lines_ptr = count_lines(input_file);
    if(*count_lines_ptr <0) {
        perror("Failed to count lines number");
        return NULL;
    }
    *dim_ptr = find_dimension(input_file);

    points_array = get_data_points_array(input_file, *dim_ptr,  *count_lines_ptr);
    return points_array;
}

int main(int argc, char** argv) {
    double **points_array = NULL;
    int dim = 0;
    int lines_num = 0;
    int i,j;
    char *file_name = argv[2];
    char *matrix_type = argv[1];
    double **matrix;
    FILE *file = fopen(file_name, "r");
    if (argc != 3) {
        perror(GENERAL_ERROR);
        return 1;
    }
    
    if (!file) {
        perror(GENERAL_ERROR);
        return 1;
    }

    points_array = parse_data(file, &dim, &lines_num);
    if (!points_array) {
        perror(GENERAL_ERROR);
        return 1;
    }
    

    matrix = malloc(lines_num * sizeof(double*));
    if (matrix == NULL) {
        perror(GENERAL_ERROR);
        return 1;
    }

    for(i=0; i<lines_num; i++) {
        matrix[i] = malloc(lines_num*sizeof(double));
        if (matrix[i] == NULL) {
            perror(GENERAL_ERROR);
            return 1;
        }
    }

    if(strcmp(matrix_type, "sym") == 0) {
        get_sym(points_array, dim, lines_num, matrix);
    } else if(strcmp(matrix_type, "ddg") == 0) {
        get_ddg(points_array, dim, lines_num, matrix);
    } else if(strcmp(matrix_type, "norm") == 0){
        get_norm(points_array, dim, lines_num, matrix);
    } else{
        perror(GENERAL_ERROR);
        return 1;
    }
    
    for (i = 0; i < lines_num; i++) {
        free(points_array[i]);
    }
    free(points_array);

    for (i = 0; i < lines_num; i++){
        for (j = 0; j < lines_num; j++) {
            printf("%.4f", matrix[i][j]);
            if(j!=lines_num-1){
                printf(",");
            }
        }
            printf("\n");
    }

    for(i=0; i<lines_num;i++) {
        free(matrix[i]);
    }
    free(matrix);
    fclose(file);
    return 0;
}