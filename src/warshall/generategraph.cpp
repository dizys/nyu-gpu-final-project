#include <stdint.h>
#include <fstream>
#include <iostream>
#include <string> 
#include <bits/stdc++.h>

using namespace std;

int main(int argc, char * argv[]){
    int n = (int) atoi(argv[1]);
    ofstream outfile;
    outfile.open("inputfile"+std::to_string(n)+".txt");
    srand(100);
    outfile << n<< endl;
    for(int i = 0; i < n; i++)
    {
        for(int j = i; j < n; j++)
        {
            if(i != j){
                int distance = rand() % 1000;
                int prob = rand() % 3;
                if(prob == 2){
                    outfile << i <<" "<< j << " "<< distance+1<< endl;
                }
            }
        }
    }
}