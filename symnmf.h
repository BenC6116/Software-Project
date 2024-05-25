//
// Created by User on 25/09/2023.
//

extern const char *GENERAL_ERROR;
double BETA = 0.5;
double EPSILON = 1e-4;
int MAX_ITER = 300;

double euc_dist_square(double*, double*, int);
void get_diag_inv_square(double **, int);
void matrix_mult(int, int, int, double**, double**, double**);
void get_sym(double**, int, int, double**);
void get_ddg(double**, int, int, double**);
void get_norm(double**, int, int, double**);
int find_dimension(FILE*);
void split_string_to_doubles(char*, double*, int);
int count_lines(FILE*);
double** get_data_points_array(FILE*, int, int);
double **parse_data(FILE*, int*, int*);