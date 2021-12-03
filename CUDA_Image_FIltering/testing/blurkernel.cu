#include"../lib/lodepng.h"
#include<iostream>
#include<time.h>
#include<assert.h>
#include<vector>
using namespace std;

#define WIDTH 800
#define HEIGHT 800

#define input_path "../images/myImage.png"
#define output_path "../images/encoded_image.png"

__global__ void blurKernel(vector<unsigned char> imgIn, vector<unsigned char>* imgOut, int w, int h)
{

}


int main(int argc, char* argv[])
{
   
#pragma region  //managing argv, argc, time
    clock_t start, end;
    int blockSize;
    assert(argc == 2);
    blockSize = atoi(argv[1]);
    assert(blockSize==8 || blockSize==16 || blockSize==32);    
#pragma endregion

#pragma region //variables declaration
    start=clock();
    dim3 dimBlock(blockSize,blockSize);
    dim3 dimGrid(ceil(WIDTH/dimBlock.x), ceil(HEIGHT/dimBlock.y));
#pragma endregion

#pragma region 
   
    vector<unsigned char> imageInput;
    vector<unsigned char> imageOutput;

    unsigned int w = WIDTH;
    unsigned int h = HEIGHT;

    lodepng::decode(imageInput, w, h, input_path);
    int size = imageInput.size() * sizeof(int);

    //unsigned char*
    //unsigned char*

    cudaMallocManaged(&imageInput, size);

    //imageOutput.resize(imageInput.size());    
    cudaMallocManaged(&imageOutput, size);

    blurKernel<<<dimGrid, dimBlock>>>(imageInput,imageOutput,WIDTH,HEIGHT);

    lodepng::encode(output_path, imageInput, w, h);

    cudaDeviceSynchronize();
#pragma endregion


#pragma region  //checking errors

#pragma endregion

#pragma region //free cuda memory and printing execution time

    //cudaFree(); 

    end=clock();
    cout << "Exe time: "<<(((double)(end-start))/CLOCKS_PER_SEC)<<" sec"<<endl;
#pragma region

    return 0;
}