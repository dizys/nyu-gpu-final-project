#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void solution(long long total, long long threads, int n, int *count, long long* queen_rows);

__device__ void process(long long index, int n, int *count, long long* queen_rows);

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
	int *count_cpu;
	int *count_gpu;
	
	
    size_t space = n*sizeof(long long);

    if(!(queen_rows_cpu = (long long *)malloc(space)))
	{
	   printf("Malloc error\n");
	   exit(1);
	}
	if( !(count_cpu = (int *)malloc(sizeof(int))) )
	{
	   printf("Malloc error\n");
	   exit(1);
	}
	count_cpu[0] = 0;

	queen_rows_cpu[0] = 1;
	for(int i = 1; i < n; i++){
		queen_rows_cpu[i] = queen_rows_cpu[i-1]*10;
	}

    start = clock();
    cudaMallocHost(&queen_rows_gpu, space);
	cudaMallocHost(&count_gpu, sizeof(int));
	

    cudaMemcpy(queen_rows_gpu, queen_rows_cpu, space, cudaMemcpyHostToDevice);
	cudaMemcpy(count_gpu, count_cpu, sizeof(int), cudaMemcpyHostToDevice);

	dim3 grid(8, 1, 1);
  	dim3 block(500, 1, 1);
	long long threads = ceil((total) / (long long)(grid.x * block.x));

	solution<<< grid, block >>>(total, threads, n, count_gpu, queen_rows_gpu);

	cudaMemcpy(count_cpu, count_gpu, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFreeHost(queen_rows_gpu); 
	cudaFreeHost(count_gpu); 
	
	end = clock();
	printf("Time used = %g sec\n", (double)(end - start) / CLOCKS_PER_SEC);

	printf("Result= %d\n", count_cpu[0]);
		
	free(queen_rows_cpu); 
	free(count_cpu); 

	return 0;
}

__global__ void solution(long long total, long long threads, int n, int *count, long long* queen_rows){
	long long i = blockDim.x * blockIdx.x + threadIdx.x;

	for (long long index = i * threads; index < (i + 1) * threads && index < total; index++)
  	{
		process(index, n, count, queen_rows);
	}
}

__device__ void process(long long index, int n, int *count, long long* queen_rows){
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
		atomicAdd(&count[0], 1);
	}
}