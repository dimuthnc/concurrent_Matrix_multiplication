#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

//take input params as the size of the square matrix
int main(int argc, char *argv[])
{
    int n;
    srand(time(NULL));
    if (argc != 2)
    {
        printf("Wrong number of input params.. Please set the N. \n");
    }
    n = atoi(argv[1]);
    //allocate memory in the heap for A, B and C matrices
    int *mat1;
    int *mat2;
    int *matR;
    int *matR_transposed;
    int *mat2_transposed;
    mat1 = (int *)malloc(n*n*sizeof(int));
    mat2 = (int *)malloc(n*n*sizeof(int));
    matR = (int *)malloc(n*n*sizeof(int));
    matR_transposed = (int *)malloc(n*n*sizeof(int));
    mat2_transposed = (int *)malloc(n*n*sizeof(int));
    double total_time = 0,transposed_total_time = 0;


    int* matR_parallel;
    int* matR_transposed_parallel;
    double total_time_parallel =0,transposed_total_time_parallel =0;

    matR_parallel =(int *)malloc(n*n*sizeof(int));
    matR_transposed_parallel = (int *)malloc(n*n*sizeof(int));
    for (int i = 0; i < n * n; i++)
    {
        mat1[i] = (int)rand() % 100;
        mat2[i] = (int)rand() % 100;
    }

    for (int iter = 0; iter < 100; iter++)
    {
        clock_t begin = clock();
        int local_sum = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    local_sum += mat1[i * n + k] * mat2[n * k+ j];
                }
                matR[i * n + j] = local_sum;
                local_sum = 0;
            }
        }
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;


        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matR_transposed[i + n * j] = mat2[i * n + j];

            }
        }

        /*
        clock_t begin_transposed = clock();
        int local_sum_tras = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    local_sum_tras += mat1[i * n + k] * mat2_transposed[j*n + k];

                }
                matR_transposed[i * n + j] = local_sum_tras;

                local_sum_tras = 0;
            }
        }
        clock_t end_transposed = clock();
        double time_spent_transposed = (double)(end_transposed - begin_transposed) / CLOCKS_PER_SEC;


        total_time += time_spent;
        transposed_total_time += transposed_total_time;
        */






        FILE *f = fopen("file.txt", "a");
        if (f == NULL)
        {
            printf("Error opening file!\n");
            exit(1);
        }
        fprintf(f, "%f,%f\n", time_spent);

        fclose(f);

        //parallel code

        clock_t begin_parallel = clock();
        int local_sum_parallel = 0;
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < n; i++)
        {
            #pragma omp parallel for schedule(runtime)
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    local_sum_parallel += mat1[i * n + k] * mat2[n * k+ j];
                }
                matR_parallel[i * n + j] = local_sum_parallel;
                local_sum_parallel = 0;
            }
        }
        clock_t end_parallel = clock();
        double time_spent_parallel = (double)(end_parallel - begin_parallel) / CLOCKS_PER_SEC;





        clock_t begin_transposed_parallel = clock();
        int local_sum_tras_parallel = 0;
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < n; i++)
        {
            #pragma omp parallel for schedule(runtime)
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    local_sum_tras_parallel += mat1[i * n + k] * mat2_transposed[j*n + k];

                }
                matR_transposed_parallel[i * n + j] = local_sum_tras_parallel;

                local_sum_tras_parallel = 0;
            }
        }
        clock_t end_transposed_parallel = clock();
        double time_spent_transposed_parallel = (double)(end_transposed_parallel - begin_transposed_parallel) / CLOCKS_PER_SEC;

        FILE *f2 = fopen("file.txt", "a");
        if (f == NULL)
        {
            printf("Error opening file!\n");
            exit(1);
        }
        fprintf(f2, "%f,%f\n", time_spent, transposed_total_time_parallel);

        fclose(f2);
        total_time_parallel += time_spent_parallel;
        transposed_total_time_parallel += transposed_total_time_parallel;







    }
    printf("Avg time for normal : %f\n", total_time / 100);
    //printf("Avg time for transposed : %f\n", transposed_total_time / 100);

    printf("Avg time for parallel : %f\n", total_time_parallel / 100);
    printf("Avg time for transposed parallel : %f\n", transposed_total_time_parallel / 100);
    







}