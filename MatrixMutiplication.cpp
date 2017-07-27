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
double sampleSize = 20;
struct timeval startTime, endTime;

void printResults(int matrix_size, double totalTime, double sqSum)
{

    double average = totalTime / sampleSize;                     //get average execution time
    double std = sqrt((sqSum / sampleSize) - (average * average)); //calculate STD
    double n = pow(((100 * 1.96 * std) / (5 * average)), 2);      //calcuate number of samples

    printf("\n \n \n \n" );

    printf("Matrix size of %d * %d \n",matrix_size, matrix_size);
    printf("Average of results %f \n",average);
    printf("Standard Deviation of results %f \n",std);
    printf("calcuated Minimum samples required (minumum n) %f \n",n);

    printf("\n \n \n \n" );
}

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

    //transpose the matrixB
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            transposed_matrixB[j][i] = matrixB[i][j];
        }
    }

    //define variables for get execution time
    double totalTime = 0, sqTime = 0;

    //loop till sample size for each matrice
    for (int count = 0; count < sampleSize; count++){

        // double begin = omp_get_wtime();         //get start time
        gettimeofday(&startTime, NULL);
//run the execution parallely
#pragma omp parallel for
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                double tot = 0;
                for (int k = 0; k < size; k++){
                    tot += matrixA[i][k] * transposed_matrixB[j][k]; //get local total
                }
                matrixC[i][j] = tot; //set value to matrixC
            }
        }
        if (count == 0){
            printf("Answer from version 1: %f\n", matrixC[1][1]);
        }

        // double end = omp_get_wtime();               //get end time
        // double time_elapsed = (end - begin);        //execution time

        gettimeofday(&endTime, NULL);
        double time_elapsed = ((endTime.tv_sec - startTime.tv_sec) * 1000000u +
                               endTime.tv_usec - startTime.tv_usec) /
                              1.e6;
        totalTime += time_elapsed;
        sqTime += (time_elapsed * time_elapsed);
    }
    printResults(size, totalTime, sqTime); //find sample size,average,STD

    /*
     * from here we implement more optimised version
     */
    //loop till sample size for each matrice
    setMatrixToZero(matrixC, size);
    totalTime = 0, sqTime = 0;
    for (int count=0; count<sampleSize; count++){
        // double begin = omp_get_wtime();
        //run execute parallely
        int thread_count = 20;
        int step = size/thread_count;
        //variables for iteration
        int i,j,k,a,b,c;
        double tot;
        //get start time
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
        if (count == 0){
            printf("Answer from version 2: %f\n", matrixC[1][1]);
        }
        double time_elapsed = ((endTime.tv_sec - startTime.tv_sec) * 1000000u +
                               endTime.tv_usec - startTime.tv_usec) /
                              1.e6;
        totalTime += time_elapsed;
        sqTime += (time_elapsed * time_elapsed);
    }
    printf("more optimized version\n");
    printResults(size, totalTime, sqTime); //find sample size,average,STD

    //free the allocated memory
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(transposed_matrixB);
}

void matrixMultiplicationSequential(int size){
    /*
    Each vector will hold the matrix in a 2D vector space
    */
    vector<vector<double> > matrixA(size, vector<double>(size));
    vector<vector<double> > matrixB(size, vector<double>(size));
    vector<vector<double> > matrixC(size, vector<double>(size));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrixA[i][j] = rand() % 100000;
            matrixB[i][j] = rand() % 100000;
        }
    }

    double totalTime = 0, sqTime = 0;

    for (int count = 0; count < sampleSize; count++)
    {

        gettimeofday(&startTime, NULL);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrixC[i][j] = 0;
                for (int k = 0; k < size; k++)
                {
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

    printResults(size,totalTime,sqTime);
    //printf("Time taken for matrix size of %d %d is %f \n",size,size,(totalTime/noOfSamples));

}
void matrixMultiplicationParallel(int size){
    /*
    Each vector will hold the matrix in a 2D vector space
    */
    vector<vector<double> > matrixA(size, vector<double>(size));
    vector<vector<double> > matrixB(size, vector<double>(size));
    vector<vector<double> > matrixC(size, vector<double>(size));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrixA[i][j] = rand() % 100000;
            matrixB[i][j] = rand() % 100000;
        }
    }

    double totalTime = 0, sqTime = 0;

    for (int count = 0; count < sampleSize; count++)
    {

        gettimeofday(&startTime, NULL);
        #pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrixC[i][j] = 0;
                for (int k = 0; k < size; k++)
                {
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

    printResults(size,totalTime,sqTime);
    //printf("Time taken for matrix size of %d %d is %f \n",size,size,(totalTime/noOfSamples));

}



int main(int argc, char **argv)
{
    for (int i = 200; i <= 2000; i += 200)
    {
        matrixMultiplicationSequential(i);
        matrixMultiplicationParallel(i);
        matrixMultiplicationOptimized(i);
    }
    return 0;
}
