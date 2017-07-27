//
// Created by Dimuth on 7/27/2017.
//

#include <cstdlib>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <vector>
using namespace std;
double sampleSize = 180;
struct timeval startTime, endTime;


//THIS METHOD IS USED TO RESET RESULTING MATRIX TO ZERO MATRIX AFTER CALCULATION
void setMatrixToZero(double **matrix, int size){
    for (int i=0; i<size ; i++){
        for (int j=0 ; j<size ; j++){
            matrix[i][j] = 0;
        }
    }
}

double **matrixInitialization(int size){
    double **matrix = (double **)malloc(sizeof(double *) * size);
    for (int i = 0; i < size; i++){
        matrix[i] = (double *)malloc(size*sizeof(double));
    }
    return matrix;
}

//THIS METHOD WILL PRINT ALL THE NECCESSARY INFORMATION ABOUT EACH MATRIX MULTIPLICAT
void printResults(int matrix_size, double totalTime, double sqSum,char* description){

    double average = totalTime / sampleSize;
    //STANDARD DEVIATION CALCULATION
    double std = sqrt((sqSum / sampleSize) - (average * average));
    //CALCULATING THE REQUIRED SAMPLE SIZE BASED ON STD
    double n = pow(((100 * 1.96 * std) / (5 * average)), 2);

    printf("\n \n" );

    printf(description);
    printf("\n");
    printf("Matrix size of %d * %d \n",matrix_size, matrix_size);
    printf("Average of results %f \n",average);
    printf("Standard Deviation of results %f \n",std);
    printf("calcuated Minimum samples required (minumum n) %f \n",n);

    printf("\n \n" );
}


//THIS METHOD EXECUTE SEQUENTIAL CALCULATION FOR MATRIX MULTIPLICATION
void matrixMultiplicationSequential(int size){

    vector<vector<double> > matrixA(size, vector<double>(size));
    vector<vector<double> > matrixB(size, vector<double>(size));
    vector<vector<double> > matrixC(size, vector<double>(size));

    //INITIALIZING MATRICES A AND B
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            matrixA[i][j] = rand() % 100000;
            matrixB[i][j] = rand() % 100000;
        }
    }
    //SEQUENTIAL CALCULATION
    double totalTime = 0, sqTime = 0;

    for (int count = 0; count < sampleSize; count++){
        gettimeofday(&startTime, NULL);
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                matrixC[i][j] = 0;
                for (int k = 0; k < size; k++){
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
        gettimeofday(&endTime, NULL);
        double time_elapsed = ((endTime.tv_sec - startTime.tv_sec) * 1000000u +
                               endTime.tv_usec - startTime.tv_usec) /
                              1.e6;
        sqTime += (time_elapsed * time_elapsed);
        totalTime += time_elapsed;
    }
    //PRINTING RESULTS
    printResults(size,totalTime,sqTime,"Sequential Results");

}
//NORMAL PARALLEL CALCULATION OF MATRIX MULTIPLICATION
void matrixMultiplicationParallel(int size){

    vector<vector<double> > matrixA(size, vector<double>(size));
    vector<vector<double> > matrixB(size, vector<double>(size));
    vector<vector<double> > matrixC(size, vector<double>(size));

    //MATRIX INITIALIZATION
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            matrixA[i][j] = rand() % 100000;
            matrixB[i][j] = rand() % 100000;
        }
    }

    double totalTime = 0, sqTime = 0;

    for (int count = 0; count < sampleSize; count++){

        gettimeofday(&startTime, NULL);
        //OPENMP CODE FOR PARALLEL FOR
        #pragma omp parallel for
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                matrixC[i][j] = 0;
                for (int k = 0; k < size; k++){
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
        gettimeofday(&endTime, NULL);
        double time_elapsed = ((endTime.tv_sec - startTime.tv_sec) * 1000000u +
                               endTime.tv_usec - startTime.tv_usec) /
                              1.e6;
        sqTime += (time_elapsed * time_elapsed);
        totalTime += time_elapsed;
    }
    //PRINTING RESULTS
    printResults(size,totalTime,sqTime,"Parallel Results");

}
////OPTIMIZED MATRIX MULTIPLICATION (TWO METHODS)
void matrixMultiplicationOptimized(int size){

    double **matrixA, **matrixB, **matrixC, **transposed_matrixB;

    matrixA = matrixInitialization(size);
    matrixB = matrixInitialization(size);
    matrixC = matrixInitialization(size);
    transposed_matrixB = matrixInitialization(size);

    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++){
            matrixA[i][j] = rand() % 10000;
            matrixB[i][j] = rand() % 10000;
            matrixC[i][j] = 0;
        }
    }

    //MATRIXB TRANSPOSE CALCULATION
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            transposed_matrixB[j][i] = matrixB[i][j];
        }
    }
    //FIRST OPTIMIZATION (TAKING THE TRANSPOSE OF MATRIXB)

    double totalTime = 0, sqTime = 0;

    for (int count = 0; count < sampleSize; count++){

        gettimeofday(&startTime, NULL);

        #pragma omp parallel for
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                double tot = 0;
                for (int k = 0; k < size; k++){
                    tot += matrixA[i][k] * transposed_matrixB[j][k];
                }
                matrixC[i][j] = tot;
            }
        }



        gettimeofday(&endTime, NULL);
        double time_elapsed = ((endTime.tv_sec - startTime.tv_sec) * 1000000u +
                               endTime.tv_usec - startTime.tv_usec) / 1.e6;
        totalTime += time_elapsed;
        sqTime += (time_elapsed * time_elapsed);
    }
    printResults(size, totalTime, sqTime,"First Optimization Results"); //find sample size,average,STD
    //SECOND OPTIMIZATION(TILED METHOD)

    setMatrixToZero(matrixC, size);
    totalTime = 0, sqTime = 0;
    for (int count=0; count<sampleSize; count++){

        int thread_count = 20;
        int step = size/thread_count;

        int i,j,k,a,b,c;
        double tot;

        gettimeofday(&startTime, NULL);
        #pragma omp parallel for shared(matrixA, matrixB, matrixC) private(i, j, k, tot) schedule(static) num_threads(thread_count)
        for (a = 0; a < size; a += step){
            for (b = 0; b < size; b += step){
                for (c = 0; c < size; c += step){
                    //from here same calculation as above optimization
                    for (i = a; i < min(size, a + step); i++){
                        for (j = b; j < min(size, b + step); j++){
                            tot = 0;
                            for (k = c; k < min(size, c + step); k++){
                                tot += matrixA[i][k] * transposed_matrixB[j][k]; //get local total
                            }
        #pragma omp critical
                            matrixC[i][j] += tot; //set value to matrixC
                        }
                    }
                }
            }
        }
        gettimeofday(&endTime, NULL);

        double time_elapsed = ((endTime.tv_sec - startTime.tv_sec) * 1000000u +
                               endTime.tv_usec - startTime.tv_usec) /
                              1.e6;
        totalTime += time_elapsed;
        sqTime += (time_elapsed * time_elapsed);

    }

    printResults(size, totalTime, sqTime,"First Optimization Results"); //find sample size,average,STD

    //free the allocated memory
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(transposed_matrixB);
}

//EXECUTION OF EACH VERSIONS FOR DIFFERENT N
int main(int argc, char **argv)
{
    for (int matSize = 400; matSize <= 2000;matSize += 200){
        matrixMultiplicationSequential(matSize);
        matrixMultiplicationParallel(matSize);
        matrixMultiplicationOptimized(matSize);
    }
    return 0;
}
