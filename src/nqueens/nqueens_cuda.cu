#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void solution(long long total, int n, bool *results, long long* queen_rows);

int main(int argc, char *argv[]){

	if(argc != 2){
		printf("Command Line: ./program n \n");
		printf("n is the size of the board \n");
		exit(1);
	}
    long long n = atoi(argv[1]);
    long long total = pow(n,n);  
        
    double start, end;
    
	long long *queen_rows_cpu;
	long long *queen_rows_gpu;
	bool *results_cpu;
	bool *results_gpu;
	
    size_t space = n*sizeof(long long);
	size_t space2 = total*sizeof(bool);

    if(!(queen_rows_cpu = (long long *)malloc(space)))
	{
	   printf("Malloc error\n");
	   exit(1);
	}
	if( !(results_cpu = (bool *)malloc(pow(n,n)*sizeof(bool))) )
	{
	   printf("Malloc error\n");
	   exit(1);
	}

	queen_rows_cpu[0] = 1;
	for(int i = 1; i < n; i++){
		queen_rows_cpu[i] = queen_rows_cpu[i-1]*10;
	}

	for(long long i = 0; i < total; i++){
		results_cpu[i] = false;
	}

    start = clock();
    cudaMallocHost(&queen_rows_gpu, space);
	cudaMallocHost(&results_gpu, space2);

    cudaMemcpy(queen_rows_gpu, queen_rows_cpu, space, cudaMemcpyHostToDevice);
	cudaMemcpy(results_gpu, results_cpu, space2, cudaMemcpyHostToDevice);

	
	long long BLOCK_SIZE = 512;
	long long BLOCKS_NUM = total/BLOCK_SIZE+1;

	solution<<< BLOCKS_NUM, BLOCK_SIZE >>>(total, n, results_gpu, queen_rows_gpu);

	cudaMemcpy(results_cpu, results_gpu, space2, cudaMemcpyDeviceToHost);

	cudaFreeHost(results_gpu); 
	cudaFreeHost(queen_rows_gpu); 
	
	end = clock();
	printf("Time used = %g sec\n", (double)(end - start) / CLOCKS_PER_SEC);

	int result = 0;
	for(long long i = 0; i < total; i++){
		if(results_cpu[i]){
			result++;
		}
	}
    printf("Result= %d\n", result);
		
	free(queen_rows_cpu); 
	free(results_cpu); 

	return 0;
}

__global__ void solution(long long total, int n, bool *results, long long* queen_rows){
	long long index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= total){
		return;
	}
	long long  current = index;
	long long  board = 0;
	for (long long i = 0; i < n; i++)
	{
        board *= 10;
		board += current % n;
		current /= n;
	}
	bool hasError = false;
	for (long long i = 0; i < n; i++)
	{
        long long iqueen = board / queen_rows[i];
		iqueen = iqueen % 10;
		for (long long j = i+1; j < n; j++)
		{
            long long jqueen = board / queen_rows[j];
		    jqueen = jqueen % 10;
			if (iqueen == jqueen|| iqueen - jqueen == i - j || iqueen - jqueen == j - i){
				hasError = true;
				break;
			}
		}
		if(hasError){
			break;
		}
	}
	if (!hasError)
	{
		results[index] = true;
	}else{
		results[index] = false;
	}
    
}