#include <gtk/gtk.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "../src/gtkfunctions.cpp"
#include "../src/hyperfunctions.cpp"
using namespace cv;
using namespace std;

  
  
// https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes 
int main (int argc, char *argv[])
{
    HyperFunctions HyperFunctions1;
    string file_name1="../../HyperImages/Public_Images/hyperspectral_images/Indian_pines.tiff";
    string file_name2="../../HyperImages/Public_Images/ground_truth/Indian_pines_gt.tiff";
    // step 1 load hyperspectral image 
    HyperFunctions1.LoadImageHyper1(file_name1);

    // step 2 load ground truth image
    Mat gt_img=imread(file_name2, IMREAD_COLOR);
    
    
    // step 3 make sure ground truth image is 8 bit single channel
    // ref https://gist.github.com/yangcha/38f2fa630e223a8546f9b48ebbb3e61a
    //cout<<gt_img.type()<<endl;
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
    
    // step 4 get number of semantic classes 
    // assumption 0 is unknown
    // class values 1-N (N is total number of classes)
    // ref https://stackoverflow.com/questions/15955305/find-maximum-value-of-a-cvmat
    // only care about max value
    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;

    minMaxLoc( gt_img, &minVal, &maxVal, &minLoc, &maxLoc );
    cout<<"Number of semantic classes: "<<maxVal<<endl;
    // step 5 get pixel coordinates of each semantic class
    // vector size not initialized properly
     vector<vector<Point>> class_coordinates((int)(maxVal+1));

    // need to initialize so right size based on result of step 4
    int temp_val;
    for  (int i=0; i<gt_img.rows ; i++)
    {
        for (int j=0; j<gt_img.cols ; j++)
        {
            temp_val=gt_img.at<uchar>(i,j);
            Point temp_point=Point(i,j);
            class_coordinates[temp_val].push_back(temp_point);  
            //cout<<temp_val<<endl;
        }
     }      
    // verify number of samples is right
    cout<<"class and number of samples in class"<<endl;
    for  (int i=0; i<class_coordinates.size() ; i++)
    {
        cout<<i<<"  "<<class_coordinates[i].size()<<endl;
    
    }
    
    
    
    // find average spectrum for each semantic class
    for  (int i=0; i<class_coordinates.size() ; i++)
    {
        // for each semantic class find the average ref spectrum 
        for  (int j=0; j<class_coordinates[i].size() ; j++)
        {
    
        }
        
    }
    
    /*
    
    get spectral values example 
    int img_hist[mlt1.size()-1];
    for (int i=0; i<=mlt1.size()-1;i++)
    {
        img_hist[i]=mlt1[i].at<uchar>(cur_loc);
    }
    
    
    */ 
    
    
    
    // extra display gt and hyperimg 
    /*Mat gt_normal;
    normalize(gt_img, gt_normal, 0,255, NORM_MINMAX, CV_8UC1);
    imshow("gt img", gt_normal);
    imshow("hyper img", HyperFunctions1.mlt1[70]);
    cv::waitKey();*/
  
  cout<<"done"<<endl;
  return 0;
}


