/**
GPGPU assignment 3: Parallel image convolutional filtering
    @file first.cpp
    @author Canonaco Martina @author Gena Davide @author Morello Michele @author Oliviero Tiziana
    @version 02 December 2021 
*/

#include "lodepng.h"
#include <iostream>
using namespace std;

#define image_path "myImage.png"
#define output_path "encoded_image.png"

void print(vector <unsigned char> const &image) {
   for(int i=0; i < image.size(); i++)
        cout << image.at(i) << ' ';
}

int main()
{
    unsigned width = 800;
    unsigned height = 800;
    vector<unsigned char> image;

    lodepng::decode(image, width, height, image_path);

    if(image.empty())
        cout<<"EMPTY :("<<endl;
    //else print(image);

    lodepng::encode(output_path, image, width, height);

    return 0;
}