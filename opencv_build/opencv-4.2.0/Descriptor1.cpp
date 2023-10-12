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


void computeCustomDescriptor ( const cv::Mat& feature_img, std::vector<cv::KeyPoint> & keypoints,cv::Mat& descriptors)
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
      feature_img.at<uchar>(cvRound (y +scale), cvRound(x + scale))
      * feature_img.at<uchar>(cvRound (y-scale), cvRound(x - scale))
      - feature_img.at<uchar> (cvRound(y + scale), cvRound (x-scale))
      * feature_img.at<uchar> (cvRound(y -scale), cvRound(x + scale ));

      descriptors.at<float>(i,j) = static_cast<float>(det_Hessian);
    }
  }

}