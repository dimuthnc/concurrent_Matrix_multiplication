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
using namespace std;
double samples_count = 1;
struct timeval startVal, endVal;

void printDetails(int size, double totTime, double sqSum)
{

    double avg = totTime / samples_count;                     //get average execution time
    double STD = sqrt((sqSum / samples_count) - (avg * avg)); //calculate STD
    double n = pow(((100 * 1.96 * STD) / (5 * avg)), 2);      //calcuate number of samples
    printf("Matrix size of %d * %d is Average=%f &  STD=%f  &   n=%f\n", size, size, avg, STD, n);
}

void setAllZero(double **matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = 0;
        }
    }
}

double **initMatrix(int size)
{
    double **matrix = (double **)malloc(sizeof(double *) * size);

    for (int i = 0; i < size; i++)
    {
        matrix[i] = (double *)malloc(sizeof(double) * size);
    }
    return matrix;
}

void matrixMultiplication(int size)
{

    double **matrixA, **matrixB, **matrixC, **matrixB_transposed;

    //initMatrix(matrixC, size);

    //assign random values
    matrixA = initMatrix(size);
    matrixB = initMatrix(size);
    matrixC = initMatrix(size);
    matrixB_transposed = initMatrix(size);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrixA[i][j] = rand() % 10000;
            matrixB[i][j] = rand() % 10000;
            matrixC[i][j] = 0;
        }
    }

    printf("started for size: %d\n", size);
    //transpose the matrixB
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrixB_transposed[j][i] = matrixB[i][j];
        }
    }

    //define variables for get execution time
    double totTime = 0, sqTime = 0;

    //loop till sample size for each matrice
    for (int count = 0; count < samples_count; count++)
    {

        // double begin = omp_get_wtime();         //get start time
        gettimeofday(&startVal, NULL);
//run the execution parallely
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                double tot = 0;
                for (int k = 0; k < size; k++)
                {
                    tot += matrixA[i][k] * matrixB_transposed[j][k]; //get local total
                }
                matrixC[i][j] = tot; //set value to matrixC
            }
        }
        if (count == 0)
        {
            printf("Answer from version 1: %f\n", matrixC[1][1]);
        }

        // double end = omp_get_wtime();               //get end time
        // double time_elapsed = (end - begin);        //execution time

        gettimeofday(&endVal, NULL);
        double time_elapsed = ((endVal.tv_sec - startVal.tv_sec) * 1000000u +
                               endVal.tv_usec - startVal.tv_usec) /
                              1.e6;
        totTime += time_elapsed;
        sqTime += (time_elapsed * time_elapsed);
    }
    printDetails(size, totTime, sqTime); //find sample size,average,STD

    /*
     * from here we implement more optimised version
     */
    //loop till sample size for each matrice
    setAllZero(matrixC, size);
    totTime = 0, sqTime = 0;
    for (int count = 0; count < samples_count; count++)
    {
        // double begin = omp_get_wtime();
        //run execute parallely
        int thread_count = 20;
        int step = size / thread_count;
        //variables for iteration
        int i, j, k, i0, j0, k0;
        double tot;
        //get start time
        gettimeofday(&startVal, NULL);
#pragma omp parallel for shared(matrixA, matrixB, matrixC) private(i, j, k, tot) schedule(static) num_threads(thread_count)
        for (i0 = 0; i0 < size; i0 += step)
        {
            for (j0 = 0; j0 < size; j0 += step)
            {
                for (k0 = 0; k0 < size; k0 += step)
                {
                    //from here same calculation as above optimization
                    for (i = i0; i < min(size, i0 + step); i++)
                    {
                        for (j = j0; j < min(size, j0 + step); j++)
                        {
                            tot = 0;
                            for (k = k0; k < min(size, k0 + step); k++)
                            {
                                tot += matrixA[i][k] * matrixB_transposed[j][k]; //get local total
                            }
#pragma omp critical
                            matrixC[i][j] += tot; //set value to matrixC
                        }
                    }
                }
            }
        }
        gettimeofday(&endVal, NULL);
        if (count == 0)
        {
            printf("Answer from version 2: %f\n", matrixC[103][1]);
        }
        double time_elapsed = ((endVal.tv_sec - startVal.tv_sec) * 1000000u +
                               endVal.tv_usec - startVal.tv_usec) /
                              1.e6;
        totTime += time_elapsed;
        sqTime += (time_elapsed * time_elapsed);
    }
    printf("more optimized version\n");
    printDetails(size, totTime, sqTime); //find sample size,average,STD

    //free the allocated memory
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(matrixB_transposed);
}

int main(int argc, char **argv)
{
    for (int i = 2000; i <= 2000; i += 200)
    {
        matrixMultiplication(i);
    }
    return 0;
}
