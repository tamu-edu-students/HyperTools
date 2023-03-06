#include <gtk/gtk.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "gtkfunctions.cpp"
#include "hyperfunctions.cpp"
using namespace cv;
using namespace std;

  
  

int main (int argc, char *argv[])
{
    HyperFunctions HyperFunctions1;
    string file_name1="../../HyperImages/Public_Images/hyperspectral_images/Indian_pines.tiff";
    string file_name2="../../HyperImages/Public_Images/ground_truth/Indian_pines_gt.tiff";
    // step 1 load hyperspectral image 
    HyperFunctions1.LoadImageHyper1(file_name1);

    // step 2 load ground truth image
    Mat gt_img=imread(file_name2, IMREAD_COLOR);
    
    
    // step 3 
    cout<<gt_img.type()<<endl;
    //cout<<gt_img<<endl;
    if (gt_img.type()==16)
    { // color img
        //gt_img.convertTo(gt_img,CV_8UC1);converts to 8bit
        cvtColor(gt_img, gt_img, COLOR_BGR2GRAY);
    
    }
    else if (gt_img.type()==0)
    {
     // right now dont do anything 
    
    }
    else
    {
        cout<<"unsupported image type"<<endl;
    }
    
        cout<<gt_img.type()<<endl;
  
    // extra display gt and hyperimg 
    /*Mat gt_normal;
    normalize(gt_img, gt_normal, 0,255, NORM_MINMAX, CV_8UC1);
    imshow("gt img", gt_normal);
    imshow("hyper img", HyperFunctions1.mlt1[70]);
    cv::waitKey();*/
  
  cout<<"done"<<endl;
  return 0;
}


