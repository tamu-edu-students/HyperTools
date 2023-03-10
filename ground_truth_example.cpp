#include <gtk/gtk.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>

#include "gtkfunctions.cpp"
#include "hyperfunctions.cpp"
using namespace cv;
using namespace std;


  
// https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes 
int main (int argc, char *argv[])
{
    HyperFunctions HyperFunctions1;
    string file_name1 = "../../HyperImages/hyperspectral_images/Indian_pines.tiff";
    string file_name2 = "../../HyperImages/ground_truth/Indian_pines_gt.tiff";
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
        cout << "unsupported image type" << endl;
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
    // 2D array?
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
        cout << i << "  " <<class_coordinates[i].size() << endl;
    
    }
    
    // find average spectrum for each semantic class
    int avgSpectrums[class_coordinates.size()][HyperFunctions1.mlt1.size()];

    for  (int i=0; i<class_coordinates.size() ; i++)    // for each class
    {
        for  (int j=0; j<HyperFunctions1.mlt1.size() ; j++)
        {
            avgSpectrums[i][j] = 0;
        }
    }
    
    for  (int i=0; i<class_coordinates.size() ; i++)    // for each class
    {
        for (int k = 0; k < class_coordinates[i].size(); k++){
            Point tempPt = class_coordinates[i][k];
            for  (int j=0; j<HyperFunctions1.mlt1.size() ; j++)
            {
                avgSpectrums[i][j] += HyperFunctions1.mlt1[j].at<uchar>(tempPt);
            }
        }

        for  (int j=0; j<HyperFunctions1.mlt1.size() ; j++)
        {
            avgSpectrums[i][j] /= class_coordinates[i].size();
            // cout << avgSpectrums[i][j] << endl;
        }
    }

    // set parameters for json file
    // set wavelength of each layer and 8bit value (0-255)
    // set semantic class name and associated number from gt image
    // write to json file 
    // reference hyperfunctions.cpp void  HyperFunctions::save_ref_spec_json(string item_name)
    
    // below is a sample script and intended to be used as a framework
    // needs to be written so numbers are ordered properly right now 1 is next to 10 instead of 2
    string spectral_database="../json/spectral_database_gt.json";

    std::ofstream file_id;
    file_id.open(spectral_database);
    Json::Value value_obj;
    // value_obj = completeJsonData2;
    
    vector<string> class_list{"Unknown", "Alfalfa", "Corn-notill", "Corn-mintill","Corn","Grass-pasture", "Grass-trees", "Grass-pasture-mowed","Hay-windrowed","Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"};
    
    
    for (int j=0; j<class_coordinates.size() ; j++)
    {
        // i should be the spectral wavelength (modify for loop)
        for (int i=0; i<=HyperFunctions1.mlt1.size();i+=1)
        {
            value_obj["Spectral_Information"][class_list[j]][to_string(i)] = avgSpectrums[j][i]; //i*10 should be value between 0-255 corresponding to reflectance

        }
        // may need to also set color information for visualization in addition to the class number 
        value_obj["Color_Information"][class_list[j]]["Class_Number"] = j;
    }
    
    Json::StyledWriter styledWriter;
    file_id << styledWriter.write(value_obj);
    file_id.close();
    
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


