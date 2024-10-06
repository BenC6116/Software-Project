
# Symmetric Non-Negative Matrix Factorization (SymNMF) Project

This project implements a clustering algorithm based on Symmetric Non-negative Matrix Factorization (SymNMF) and compares its performance to the K-means algorithm. The implementation includes a combination of C and Python code, with C modules wrapped as Python extensions.

## Project Structure

- **symnmf.py**: Python interface for SymNMF algorithm.
- **symnmf.h**: Header file defining functions used in the C implementation.
- **symnmf.c**: C implementation of the SymNMF algorithm and auxiliary matrix operations.
- **symnmfmodule.c**: Python C API wrapper for integrating the C implementation into Python.
- **analysis.py**: Python script to compare the performance of SymNMF and K-means using the silhouette score.
- **setup.py**: Python setup file for building the C extension module.
- **Makefile**: Makefile to compile the C implementation into an executable.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.x
- GCC (for compiling the C code)
- Required Python packages: `numpy`, `pandas`, `scikit-learn`

To install the Python packages, run:
```bash
pip install numpy pandas scikit-learn
```

## Building the Project

### Option 1: Using the Makefile
To compile the C code into an executable, run:
```bash
make
```

This will create the `symnmf` executable.

To clean the build:
```bash
make clean
```

### Option 2: Using Python Setup Script
You can also build the Python C extension using the setup script. Run the following command in the project directory:
```bash
python3 setup.py build_ext --inplace
```

This will create the `mysymnmf.so` shared object file, which can be imported and used in Python.

## Running the Program

### SymNMF Python Script
To run the SymNMF algorithm or generate specific matrix outputs using the Python script, use the following command format:

```bash
python3 symnmf.py <k> <goal> <file_name>
```

- `k`: Number of clusters.
- `goal`: The operation to perform:
  - `symnmf`: Perform the full SymNMF algorithm and output the decomposition matrix `H`.
  - `sym`: Calculate and output the similarity matrix.
  - `ddg`: Calculate and output the diagonal degree matrix.
  - `norm`: Calculate and output the normalized similarity matrix.
- `file_name`: Path to the input file containing the dataset.

Example:
```bash
python3 symnmf.py 3 symnmf input_data.txt
```

### C Implementation
You can also run the C version of the program directly from the command line:

```bash
./symnmf <goal> <file_name>
```

- `goal`: The same goals as described above (`sym`, `ddg`, or `norm`).
- `file_name`: Path to the input file.

Example:
```bash
./symnmf sym input_data.txt
```

### Analysis Script
The analysis script compares the performance of SymNMF and K-means on a given dataset using the silhouette score. Run it with:

```bash
python3 analysis.py <k> <file_name>
```

Example:
```bash
python3 analysis.py 3 input_data.txt
```

This will output the silhouette scores for both SymNMF and K-means.

## Input Format

The input file should be a `.txt` file containing the data points, with each row representing a data point, and each value separated by commas. There should be no header.

Example:
```
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
```

## Assumptions

1. The number of clusters `k` must be greater than 1 and less than the number of data points.
2. Outputs are formatted to 4 decimal places.
3. The program assumes that all given data points are unique.
4. The C code must compile cleanly with no errors or warnings using the provided Makefile.
5. Errors are handled by printing "An Error Has Occurred" and terminating the program.

## Example Usage

1. To run the SymNMF algorithm and output the resulting matrix `H`:
   ```bash
   python3 symnmf.py 3 symnmf input_data.txt
   ```

2. To compute the similarity matrix using the C program:
   ```bash
   ./symnmf sym input_data.txt
   ```

3. To compare SymNMF and K-means clustering on a dataset:
   ```bash
   python3 analysis.py 3 input_data.txt
   ```

## Cleaning Up

After building and running the project, you can clean up the generated files using:
```bash
make clean
```

## References

The algorithm is based on the following research paper:
- Da Kuang, Chris Ding, and Haesun Park. *Symmetric nonnegative matrix factorization for graph clustering*. In Proceedings of the 2012 SIAM International Conference on Data Mining (SDM), Proceedings, pages 106–117. Society for Industrial and Applied Mathematics, April 2012.
