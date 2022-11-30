#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <stdint.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <string> 
#include <bits/stdc++.h>

using namespace std;
__global__ void Warshall_kernel(int n, int i,long long* graph_gpu);

int main(int argc, char * argv[]){
    int n;
    ifstream infile; 
    infile.open(argv[1]); 
    infile >> n;
    int size = n*n;
    long long *graph=(long long *)malloc(size * sizeof(long long));

    for(int i = 0; i < n; i++)
    {
        for(int j = i; j < n; j++)
        {
            if(i == j){
                graph[i*n+j] = 0;
            }else{
                graph[i*n+j] = INT_MAX;
                graph[j*n+i] = INT_MAX;
            }
        }
    }
    
    int a, b;
    long long distance;
    while (infile >> a >> b >> distance)
    {
        graph[a*n+b] = distance;
        graph[b*n+a] = distance; 
    }
    infile.close();

    long long *graph_gpu;
    cudaMalloc((void **)&graph_gpu, size * sizeof(long long));
    cudaMemcpy(graph_gpu, graph, size * sizeof(long long), cudaMemcpyHostToDevice);

    double start, end;
    start = clock();

    long long BLOCK_SIZE = 512;
	long long BLOCKS_NUM = n/BLOCK_SIZE+1;

    for(int i=0; i<n; i++)
    {
        std::cout<<i<<std::endl;
        Warshall_kernel<<< BLOCKS_NUM, BLOCK_SIZE >>>(n, i, graph_gpu);
        // Warshall_kernel<<< grid, block >>>(n, i, graph_gpu);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(graph, graph_gpu, size * sizeof(long long), cudaMemcpyDeviceToHost);
	cudaFreeHost(graph_gpu); 

    end = clock();
	printf("Time used = %g sec\n", (double)(end - start) / CLOCKS_PER_SEC);
    
    ofstream outfile;
    outfile.open("outputfile_cuda"+std::to_string(n)+".txt");
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            outfile<<graph[i*n+j]<<" ";
        }
        outfile<<std::endl;
    }
    free(graph);
    outfile.close();

}


__global__ void Warshall_kernel(int n, int i,long long* graph_gpu){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= n){
		return;
	}

    for(int k=0; k<n; k++)
    {
		int current=index*n+k;
        long long temp = graph_gpu[index*n+i]+graph_gpu[i*n+k];
        if (graph_gpu[current] > temp) {
            graph_gpu[current] = temp;
        }
    }
}