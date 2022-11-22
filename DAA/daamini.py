"""
    Problem Statement :- Write a program to implement matrix multiplication. 
                        Also implement multithreaded matrix multiplication with either one thread per row or one thread per cell.
                        Analyze and compare their performance.
"""

import threading
import numpy as np
import time

Matrix_A = []
Matrix_B = []
Matrix_C = []

def Input_for_matrix_dimensions():
    """
    Function to take the dimensional Input from the User.
    """
    global A_row, A_col, B_row, B_col
    global num_of_threads
    global V
    
    A_row = int(input("Enter the number of rows in first matrix: "))
    A_col = int(input("Enter the number of cloumns in first matrix: "))
    B_row = int(input("Enter the number of rows in second matrix: "))
    B_col = int(input("Enter the number of columns in second matrix: "))
    V = int(input("Do you want to see the verbose? (1 for yes, 0 for no): "))

    # number of threads = number of rows in solution matrix C = number of rows in matrix A as dim(C) = (A_row, B_col)
    num_of_threads = A_row

def Initialize_Matrix():
    """
    Funtion to initialize the matrices by populating it with random numbers from 1 to 10.
    """
    global Matrix_A
    global Matrix_B
    global Matrix_C

    if A_col == B_row:
        Matrix_A = np.random.randint(10, size=(A_row, A_col))
        
        Matrix_B = np.random.randint(10, size=(B_row, B_col))
        
        Matrix_C = np.zeros((A_row, B_col)).astype(int)
    else:
        exit()

def Matrix_multiply_parallel(row, z):
    """
    Performs multiplication with respect to a row thread.
    """
    global Matrix_A
    global Matrix_B
    global Matrix_C

    for i in range(B_col):
        for j in range(A_col):
            Matrix_C[z][i] += int(row[j] * Matrix_B[j][i])

def Thread_function():
    """
    Function for creation and execution of the row threads.
    """
    global num_of_threads
    threads = []

    # Creation and execution of threads
    for j in range(0,num_of_threads):
        t = threading.Thread(target = Matrix_multiply_parallel, args=(Matrix_A[j], j))
        threads.append(t)
        t.start()

    # Completion of the threads
    for t in threads:   
        t.join()
            
if __name__=="__main__":
    """
    Driver Code
    """
    
    # The multithreaded approach that provides the complexity of O(n^2.8074)
    Input_for_matrix_dimensions()
    Initialize_Matrix()

    start_time = time.time()
    Thread_function()
    end_time = time.time()
    
    print("Time taken to multiply two matrices in parallel comes out to be :",round(end_time - start_time, 5),"seconds\n")


    # The non thread approach that provides the time complexity of O(n^3).
    result = np.zeros((A_row, B_col)).astype(int)
    stat_time = time.time()
    # iterating by row of A
    for i in range(A_row):

        # iterating by column by B
        for j in range(B_col):

            # iterating by rows of B
            for k in range(B_row):
                result[i][j] += Matrix_A[i][k] * Matrix_B[k][j]
    end_time = time.time()

    print("Time taken to multiply two matrices comes out to be :",round(end_time - start_time, 5),"seconds\n")

    if V and result.all()==Matrix_C.all():
        print(Matrix_C,"\n")
        print(Matrix_A,"\n")
        print(Matrix_B,"\n")
