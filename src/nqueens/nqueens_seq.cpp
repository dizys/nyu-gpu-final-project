#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>

#include <omp.h>

#define MAX_N 16

int main(int argc, char* argv[])
{
	if(argc != 2){
		std::cout<<"Command Line: ./program n"<<std::endl;
		std::cout<<"n is the size of the board"<<std::endl;
		exit(1);
	}
    long long n = atoi(argv[1]);
    long long total = pow(n,n);  
        
    double start, end;
    int result = 0;
    
	long long queen_rows[n];
	queen_rows[0] = 1;
	for(int i = 1; i < n; i++){
		queen_rows[i] = queen_rows[i-1]*10;
	}
    start = omp_get_wtime();

	for (long long index = 0; index < total; index++)
	{
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
				if (iqueen == jqueen || iqueen - jqueen == i - j || iqueen - jqueen == j - i){
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
			result++;
		}
	}

	
    end = omp_get_wtime();
    printf("Time used = %g sec\n", end - start);
    printf("Result= %d\n", result);
    
	return 0;
}