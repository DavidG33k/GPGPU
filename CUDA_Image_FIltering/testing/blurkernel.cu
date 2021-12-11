#include"../lib/lodepng.h"
#include<iostream>
#include<time.h>
#include<assert.h>
#include<vector>
using namespace std;

#define WIDTH 800
#define HEIGHT 800
#define BLUR_SIZE 20
#define NUM_CHANNELS 3
#define R 0
#define G 1
#define B 2

#define input_path "../images/myImage.png"
#define output_path "../images/encoded_image.png"

__global__ void blurKernel(unsigned char* in, unsigned char* out, int w, int h, int num_channels, int channel)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if(Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;

        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                if(curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w * num_channels + curCol * num_channels + channel];
                    pixels++;
                }
            }
        }

        out[Row * w * num_channels + Col * num_channels + channel] = (unsigned char)(pixVal / pixels);
    }
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

    lodepng::decode(imageInput, w, h, input_path, LCT_RGB);
    cout << "buffer input size: " << imageInput.size() << endl;
    
    int size = imageInput.size() * sizeof(unsigned char);

    unsigned char* in;
    unsigned char* out;

    cudaMallocManaged(&in, size);    
    cudaMallocManaged(&out, size);

    for(int i=0; i<imageInput.size(); i++)
        in[i] = imageInput[i];

    cout << "ci sono" << endl;

    blurKernel<<<dimGrid, dimBlock>>>(in, out, w, h, NUM_CHANNELS, R);
    blurKernel<<<dimGrid, dimBlock>>>(in, out, w, h, NUM_CHANNELS, G);
    blurKernel<<<dimGrid, dimBlock>>>(in, out, w, h, NUM_CHANNELS, B);

    cudaDeviceSynchronize();

    cout << "ci sono" << endl;

    for(int i=0; i<imageInput.size(); i++)
        imageOutput.push_back(out[i]); //out è grande 8 ma dovrebbe essere grande quanto imageInput.size() :(

    cout << imageInput.size() << endl;

    cout << "ci sono" << endl;

    lodepng::encode(output_path, imageOutput, w, h, LCT_RGB);

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