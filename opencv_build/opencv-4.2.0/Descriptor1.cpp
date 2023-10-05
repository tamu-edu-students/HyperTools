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
// usinig hessian blob integer approximation 
    for (int j = 0; j <descriptorSize; ++j)
    {
      float scale = 1.0f + (j -descriptorSize/2) * 0.1f;

      int det_Hessian =  
      img.at<uchar>(cvRound (y +scale), cvRound(x + scale))
      * img.at<uchar>(cvRound (y-scale), cvRound(x - scale))
      - img.at<uchar> (cvRound(y + scale), cvRound (x-scale))
      * img.at<uchar> (cvRound(y -scale), cvRound(x + scale ));

      descriptors.at<float>(i,j) = static_cast<float>(det_Hessian);
    }
  }

}


int main()
{
 //initializing SURF detector
  int minHessian = 400;
  Ptr<SURF> detector = SURF::create(minHessian);

  cv::Mat img1 = cv::imread("Hyperspectral_research/Hyperimages/image.tiff", IMREAD_COLOR);
  cv:: Mat img2 = img1;
  cv::imshow ("image1",img1);

 // cv::cvtColor(img1,cv::COLOR_BGR2GRAY);


  std::vector<cv::KeyPoint> keypoint1,keypoint2;
  cv::Mat descriptors1, descriptors2;

 

  detector-> detect(img1,keypoint1,descriptors1);
  detector-> detect (img1,keypoint2,descriptors2);
 
  computeCustomDescriptor(img1,keypoint1,descriptors1);
  computeCustomDescriptor(img2,keypoint2,descriptors2);

  std::cout << "Custom Descriptors:\n" <<descriptors1<< "\n";

 
  
  /* Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
  std::vector<DMatch> matches;
  matcher -> match (descriptors1,descriptors2,matches);

  cv::Mat img_matches;
  drawMatches(img1,keypoint1,img2,keypoint2,matches,img_matches);
  imshow("Mathces",img_matches);


  waitKey();
  return 0;*/
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

