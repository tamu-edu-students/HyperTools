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
    //string file_name1 = "../../../HyperImages/hyperspectral_images/Indian_pines.tiff";
    //string file_name2 = "../../../HyperImages/ground_truth/Indian_pines_gt.tiff";
    string file_name1 = "../../../HyperImages/hyperspectral_images/Pavia.tiff";
    string file_name2 = "../../../HyperImages/ground_truth/Pavia_gt.tiff";
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
    //cout<<gt_img.type()<<endl;
    
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
    cout << "class and number of samples in class" << endl;
    for  (int i=0; i<class_coordinates.size() ; i++)
    {
        //cout << i << "  " <<class_coordinates[i].size() << endl;
    
    }
    
    // find average spectrum for each semantic class
    int class_coordinates_size = class_coordinates.size();
    int mlt1_size = HyperFunctions1.mlt1.size();
    int avgSpectrums[class_coordinates_size][mlt1_size];

    vector<vector<int>> avgSpectrums_vector ((int)(class_coordinates_size));
    for  (int i=0; i<class_coordinates.size() ; i++)    // for each class
    {
        avgSpectrums_vector[i]= vector<int> (mlt1_size);    
    }
 
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
            if (avgSpectrums[i][j]<0){ avgSpectrums[i][j]=0;}
            if (avgSpectrums[i][j]>255){ avgSpectrums[i][j]=255;}
            // cout << avgSpectrums[i][j] << endl;
            avgSpectrums_vector[i][j]=avgSpectrums[i][j];
        }
    }



   // HyperFunctions1.reference_spectrums = avgSpectrums_vector;

    vector<double> accuracy_by_class (class_coordinates.size());
    

    
    
  
   

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
        for (int i=0; i<HyperFunctions1.mlt1.size();i+=1)
        {
            string zero_pad_result;
        
            if (i<10)
            {
                zero_pad_result="000"+to_string(i);
            }
            else if(i<100)
            {
                zero_pad_result="00"+to_string(i);
            }
            else if(i<1000)
            {
                zero_pad_result="0"+to_string(i);
            }
            else if (i<10000)
            {
                zero_pad_result=to_string(i);
            }
            else
            {
                cout<<" error: out of limit for spectral wavelength"<<endl;
                return -1;
            }
            if (avgSpectrums[j][i]<0){ avgSpectrums[j][i]=0;}
            if (avgSpectrums[j][i]>255){ avgSpectrums[j][i]=255;}
            value_obj["Spectral_Information"][class_list[j]][zero_pad_result] = avgSpectrums[j][i]; //value between 0-255 corresponding to reflectance

        }

        // may need to also set color information for visualization in addition to the class number 
        //value_obj["Color_Information"][class_list[j]]["Class_Number"] = j;
        value_obj["Color_Information"][class_list[j]]["red_value"] = j;
        value_obj["Color_Information"][class_list[j]]["green_value"] = j;
        value_obj["Color_Information"][class_list[j]]["blue_value"] = j;
    }

    Json::StyledWriter styledWriter;
    file_id << styledWriter.write(value_obj);
    file_id.close();
    
    // read the json file that was just written so everything is in correct data structure
    
  /*  // below is the example database
    HyperFunctions1.read_ref_spec_json(HyperFunctions1.spectral_database);
    //This could be moved within another for loop but for debugging purposes it is separated.
    cout << "Comparing GT to SAM algorithm" << endl;
    HyperFunctions1.spec_sim_alg = 0;
    HyperFunctions1.SpecSimilParent();
    HyperFunctions1.DispSpecSim();
    HyperFunctions1.SemanticSegmenter();
    HyperFunctions1.DispClassifiedImage();
    cv::waitKey();
    */
    // below is the database that was just created
    HyperFunctions1.read_ref_spec_json(spectral_database);
    cout << "Comparing new database" << endl;
    HyperFunctions1.spec_sim_alg = 0;
    // error in below functions now sure where error comes from. need to debug
    //HyperFunctions1.SpecSimilParent();
    //HyperFunctions1.DispSpecSim();
    HyperFunctions1.SemanticSegmenter();
   // HyperFunctions1.DispClassifiedImage();
    cv::waitKey();
    

    //HyperFunctions1.SemanticSegmenter();
    cout << "SAM_img done" << endl;
    for  (int i=0; i<class_coordinates.size() ; i++)    // for each class
    {
        for (int k = 0; k < class_coordinates[i].size(); k++){
            Point tempPt = class_coordinates[i][k];
            int gtClass = i;    // assuming also comparing unknowns(0)
            int hyperfuncClass = HyperFunctions1.classified_img.at<uchar>(tempPt.x,tempPt.y);
            if (gtClass == hyperfuncClass) {
                accuracy_by_class[i] += 1;
            }
        }
        cout << "Accuracy of class: " << i << ":" << (double)(accuracy_by_class[i] ) <<"/ "<<class_coordinates[i].size()<< endl;
    }

    //
    //
    // SAM "ground truth"

    Mat SAM_img_Classified;
    SAM_img_Classified = HyperFunctions1.classified_img;

    HyperFunctions1.spec_sim_alg = 1; // Sets it to SCM
    HyperFunctions1.SemanticSegmenter();

    //For each pixel in the SCM classified image, compare to the SAM one
    vector<double> accuracy_by_class2 (class_coordinates.size());
    vector<double> SAM_in_class (class_coordinates.size()); //How many pixels the SAM said were in the class
    vector<double> SCM_in_class (class_coordinates.size()); //How many pixels the SAM said were in the class
    cout << "SCM_img done" << endl;
    for  (int i=0; i<HyperFunctions1.classified_img.rows ; i++)
    {
        for (int k = 0; k < HyperFunctions1.classified_img.cols; k++){
            int SAM_Class = SAM_img_Classified.at<uchar>(i,k);    // assuming also comparing unknowns(0)
            //cout << tempPt.x << "," << tempPt.y << endl;
            int SCM_Class = HyperFunctions1.classified_img.at<uchar>(i,k);
            //cout << gtClass << " " << hyperfuncClass << endl;
            if (SAM_Class == SCM_Class) {
                accuracy_by_class2[SAM_Class] += 1;
            }
            SAM_in_class[SAM_Class] += 1;
            SCM_in_class[SCM_Class] += 1;
        }
        
    }

    for (int i = 0; i < class_coordinates_size; i++) {
        cout << "SCM percentage agreement with SAM class " << i << ":" << 100*(double)(accuracy_by_class2[i] ) / SAM_in_class[i]<< "%" <<endl;
        cout << "SAM percentage agreement with SCM class " << i << ": " << 100*(double)(accuracy_by_class2[i] ) / SCM_in_class[i]<< "%" <<endl;
    }

    //
    //
    //
   
   /*
    int ptX = 25;
    int ptY = 49;
    cout << ptX << "," << ptY << endl;
    int ptClass = HyperFunctions1.classified_img.at<uchar>(ptX, ptY);
    cout << "HyperFunctions guesses class:" << ptClass << endl;*/
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