#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;



int main()
{
 //initializing SURF detector
  int minHessian = 400;
  //cv::Ptr<SURF> detector = SURF::create(minHessian);
  Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);

  cv::Mat img1 = cv::imread("img1.tiff", cv::IMREAD_COLOR);
  // using the img.tiff as the image that we would be taking data points from

  std::vector<cv::KeyPoint> key1 , key2;
  cv::Mat descriptors1, descriptors2;
// creating vector for the keypoints that will be detected from image
// getting variables descriptor from opencv and naming them. 

// Using the detector to detect points to be stored into a array. That wil be made into a 1D array
 
  detector -> detectAndCompute (img1, cv::noArray(),key1 , descriptors1);
  detector -> detectAndCompute (img1, cv::noArray(),key1 , descriptors2);
  std::vector<float> descriptor1D; // vector values of the descriptor changed to flaots
  //std::vector<float> descriptor1D2;
// making the descriptors into 1D descriptors

  for (const cv::KeyPoint & point: key1)
  {
    int x = static_cast<int>(point.pt.x); // takes the x vallues of the keypoints stores it in the new varaible x same for the y 
    int y = static_cast<int>(point.pt.y);
    float descriptor_val = descriptors1.at<float>(y,x); // gives us access to the keypoint locations of the descriptor.
  }


  

}