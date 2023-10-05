#include <iostream>
#include "opencv2/opencv.hpp"
//#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
//using std::end1;


void computeCustomDescriptor ( const cv::Mat& img, std::vector<cv::KeyPoint> & keypoints,cv::Mat& descriptors)
{
  int descriptorSize = 128;

  //create descriptor matrix

  descriptors = cv::Mat(keypoints.size(),descriptorSize, CV_32F);

  for ( size_t i = 0; i < keypoints.size(); ++i)
  {
    float x = keypoints[i].pt.x;
    float y = keypoints[i].pt.y;


    for (int j = 0; j <descriptorSize; ++j)
    {
      descriptors.at<float>(i,j) = x*0.1 + y* 0.5 + j*0.2;
    }
  }

}


int main()
{
 //initializing SURF detector
  int minHessian = 400;
  Ptr<SURF> detector = SURF::create(minHessian);

  cv::Mat img1 = cv::imread("img1.tiff", cv::IMREAD_COLOR);
  cv:: Mat img2 = img1;


  std::vector<cv::KeyPoint> keypoint1,keypoint2;
  cv::Mat descriptors1, descriptors2;

  detector-> detect(img1,keypoint1,descriptors1);
  detector-> detect (img1,keypoint2,descriptors2);
 
  computeCustomDescriptor(img1,keypoint1,descriptors1);
  computeCustomDescriptor(img2,keypoint2,descriptors2);
  
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
  std::vector<DMatch> matches;
  matcher -> match (descriptors1,descriptors2,matches);

  cv::Mat img_matches;
  drawMatches(img1,keypoint1,img2,keypoint2,matches,img_matches);
  imshow("Mathces",img_matches);


  waitKey();
  return 0;
}

/*else
int main ()
{
  std::cout <<"This tutorial code needs the xfeatures2d contib module to be run" << std::end1;
  return 0;
}
#endif */




// creating vector for the keypoints that will be detected from image
// getting variables descriptor from opencv and naming them. 



  //computing / creating descriptors

