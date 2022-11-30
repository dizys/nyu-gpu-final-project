#include <stdint.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <string> 
#include <bits/stdc++.h>

using namespace std;

int main(int argc, char * argv[]){
    int n;
    ifstream infile; 
    infile.open(argv[1]); 
    infile >> n;
    int size = n*n;
    long long *graph=(long long *)calloc(size, sizeof(long long));
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

    double start, end;
    start = clock();
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            for(int k=0; k<n; k++){
		        int index=j*n+k;
                long long temp = graph[j*n+i]+graph[i*n+k];
                if (graph[index] > temp) {
                    graph[index] = temp;
                }
            }
        }
    }

    end = clock();
	printf("Time used = %g sec\n", (double)(end - start) / CLOCKS_PER_SEC);
    
    ofstream outfile;
    outfile.open("outputfile_seq"+std::to_string(n)+".txt");
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