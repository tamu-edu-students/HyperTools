#include <gtk/gtk.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "gdal/gdal.h"
#include "gdal/gdal_priv.h"
#include "gdal/cpl_conv.h"  // for CPLMalloc()
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include "../src/gtkfunctions.cpp"
#include "../src/hyperfunctions.cpp"
using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace Json;

int main (int argc, char *argv[])
{
    //Collect command line arguments to determine which algorithms to use on the image.
    int firstAlgorithm = 0;
    int secondAlgorithm = 1;

    if (argc == 3) {
        // Convert command line argument to integer
        firstAlgorithm = std::stoi(argv[1]);
        secondAlgorithm = std::stoi(argv[2]);
    }
    else if (argc == 2) {
        firstAlgorithm = std::stoi(argv[1]);
    }


    //ENVI is stored as a pair of DAT and HDR
    const char* dat_file = "../../HyperImages/test.dat";

    //Ground truth png
    string gt_file = "../../HyperImages/labeledtest.png";
    
    //name of spectral database being created
    string spectral_database="../json/envi_spectral_database.json";

    // name of semantic classes 
    //vector<string> class_list{"Unknown","Alfalfa", "Corn-notill", "Corn-mintill","Corn","Grass-pasture", "Grass-trees", "Grass-pasture-mowed","Hay-windrowed","Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"};
        
    // Read the JSON file
    ifstream ifs("../../HyperImages/label_info.json");
    if (!ifs.is_open()) {
        cerr << "Error opening JSON file." << endl;
        return -1;
    }

    // Parse the JSON
    CharReaderBuilder readerBuilder;
    Value jsonData;
    parseFromStream(readerBuilder, ifs, &jsonData, nullptr);

    // Extract class names and color hex codes
    vector<string> class_list;
    vector<pair<string, int>> colorIndexVector;

    const Value& items = jsonData["items"];
    int i = 0;
    class_list.push_back("unknown");
    for (const auto& item : items) {
        string name = item["name"].asString();
        string colorHexCode = item["color_hex_code"].asString();

        class_list.push_back(name);
        colorIndexVector.push_back({colorHexCode, i});
        i++; //Rewriting this loop to look better might be a better idea
    }

    // Print the results
    //cout << "Class List:" << endl;
    for (const auto& className : class_list) {
        //cout << className << endl;
    }

   // cout << "\nColor Index Vector:" << endl;
    for (const auto& pair : colorIndexVector) {
        //cout << "Color: " << pair.first << ", Index: " << pair.second << endl;
    }

    HyperFunctions HyperFunctions1;
    // load hyperspectral image
    //HyperFunctions1.mlt1 = imageBands;
    HyperFunctions1.LoadImageHyper(dat_file);


    //std::cout << "Type of myMat: " << HyperFunctions1.mlt1[0].type() << std::endl;

    // load ground truth image
    Mat gt_img = imread(gt_file, IMREAD_COLOR);
    
    int numClasses = colorIndexVector.size(); //should read from json
    
     // Example data structure: vector of pairs (hex color, index)
    //vector<pair<string, int>> colorIndexVector;

    // get pixel coordinates of each semantic class
    // assumes we do not care about unknown and unknown =0 
    vector<vector<Point>> class_coordinates((int)(numClasses+1));

    Vec3b temp_val;// Access the pixel value at the specified point

    int which_class = 0;

    for  (int i=0; i<gt_img.rows ; i++)
    {
        for (int j=0; j<gt_img.cols ; j++)
        {
            //cout << i << j << endl;
            which_class = 0;
            temp_val=gt_img.at<Vec3b>(i,j);
            int r = temp_val[2];
            int g = temp_val[1];
            int b = temp_val[0];

            //Creates a hexadecimal color based on the rgb values in the pixel.
            ostringstream hexColor;
            hexColor << "#" << std::setfill('0') << std::setw(2) << std::hex << r
             << std::setfill('0') << std::setw(2) << std::hex << g
             << std::setfill('0') << std::setw(2) << std::hex << b;
            string targetHexColor = hexColor.str();

            //std::cout << targetHexColor << std::endl;

            //checks if the hexadecimal color of a particular pixel in the ground truth image is contained in the colorIndexVector
            //"it" is an iterator. If it finds the hex value then it will be located at that spot in the vector. If not, then it will be at the end.
            auto it = std::find_if(colorIndexVector.begin(), colorIndexVector.end(),
                                    [targetHexColor](const auto& pair) {
                                        return pair.first == targetHexColor;
                                    });

            if (it != colorIndexVector.end()) {
                which_class = it->second + 1; //the first color index corresponds to the second class index (1)
            } else {
                which_class = 0; //unknown class
                std::cout << "Hex color value not found in the data structure." << std::endl;
            }

            Point temp_point=Point(i,j);
            class_coordinates[which_class].push_back(temp_point); 
            /*
            if (temp_val>0 && temp_val<=maxVal)
            {
            	class_coordinates[which_class].push_back(temp_point);  
            }*/
        }
    }

    // verify number of firstAlgples is right
    /*cout << "class and number of firstAlgples in class" << endl;
    for  (int i=1; i<class_coordinates.size() ; i++)
    {
        cout << i << "  " <<class_coordinates[i].size() << endl;
    
    }*/

    // find average spectrum for each semantic class
    int class_coordinates_size = class_coordinates.size();
    int mlt1_size = HyperFunctions1.mlt1.size();
    int avgSpectrums[class_coordinates_size][mlt1_size];
    
    // initialize vector size 
    vector<vector<int>> avgSpectrums_vector ((int)(class_coordinates_size));
    for  (int i=1; i<class_coordinates.size() ; i++)    // for each class
    {
        avgSpectrums_vector[i]= vector<int> (mlt1_size);    
    }
    
    // initialize all values at zero 
    for  (int i=1; i<class_coordinates.size() ; i++)    // for each class
    {
        for  (int j=0; j<HyperFunctions1.mlt1.size() ; j++)
        {
            avgSpectrums[i][j] = 0;
        }
    }



    for  (int i=1; i<class_coordinates.size() ; i++)    // for each class
    {
        // for each pixel in class
        for (int k = 0; k < class_coordinates[i].size(); k++){
            Point tempPt = class_coordinates[i][k];
            // for spectrum of each pixel
            for  (int j=0; j<HyperFunctions1.mlt1.size() ; j++)
            {
                //cout << HyperFunctions1.mlt1[0].at<uchar>(tempPt) << endl;
               // cout << tempPt << endl;
                //cout << "mlt1[0] size: " << HyperFunctions1.mlt1[0].size() << endl;
                //cout << i << " " << k << " " << j << endl;
                avgSpectrums[i][j] += HyperFunctions1.mlt1[j].at<uchar>(tempPt);
            }
        }

        for  (int j=0; j<HyperFunctions1.mlt1.size() ; j++)
        {
            if (class_coordinates[i].size() > 0) {
                avgSpectrums[i][j] /= class_coordinates[i].size();
            }
            if (avgSpectrums[i][j]<0){ avgSpectrums[i][j]=0;}
            if (avgSpectrums[i][j]>255){ avgSpectrums[i][j]=255;}
            avgSpectrums_vector[i][j]=avgSpectrums[i][j];
        }
    }


    //int al = sizeof(avgSpectrums)/sizeof(avgSpectrums[0]); //length calculation
   //cout << "The length of the array is: " << al << endl;
   //cout << "The length of the vector is: " << avgSpectrums_vector.size() << endl;

//cout << avgSpectrums_vector[44][0] << endl;

    vector<double> accuracy_by_class (class_coordinates.size());
    


    std::ofstream file_id;
    file_id.open(spectral_database);
    Json::Value value_obj;
    // value_obj = completeJsonData2;
    
    
    for (int j=1; j<class_coordinates.size() ; j++)
    {
        // i should be the spectral wavelength (modify for loop)
        // want to save by wavelength value rather than the layer value
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
            //cout << i << " " << j << endl;
            if (avgSpectrums[j][i]<0){ avgSpectrums[j][i]=0;}
            if (avgSpectrums[j][i]>255){ avgSpectrums[j][i]=255;}
            value_obj["Spectral_Information"][class_list[j]][zero_pad_result] = avgSpectrums_vector[j][i]; //value between 0-255 corresponding to reflectance
            //cout << i << " " << j << "time2" << endl;
        }

        //Guessing seg fault
        //cout << "line 245" << endl;
        // may need to also set color information for visualization in addition to the class number 
        //value_obj["Color_Information"][class_list[j]]["Class_Number"] = j;
        value_obj["Color_Information"][class_list[j]]["red_value"] = j;
        value_obj["Color_Information"][class_list[j]]["green_value"] = j;
        value_obj["Color_Information"][class_list[j]]["blue_value"] = j;

        //cout << "line 252" << endl;
    }


    Json::StyledWriter styledWriter;
    file_id << styledWriter.write(value_obj);
    file_id.close();
    
    cout<<"spectral database was saved to json file"<<endl;

    // read the json file that was just written so everything is in correct data structure
    HyperFunctions1.read_ref_spec_json(spectral_database);
    
    // show spectral similarity image 
    /*
    HyperFunctions1.spec_sim_alg = 0;
    HyperFunctions1.SpecSimilParent();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();*/
    

    // set alg to firstAlg and perform semantic segmentation
    //cout << "Segmenting Image with firstAlg algorithm" << endl;
    HyperFunctions1.spec_sim_alg = firstAlgorithm;
    auto start = high_resolution_clock::now();
    HyperFunctions1.SemanticSegmenter();
    auto end = high_resolution_clock::now();
     cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    //HyperFunctions1.DispClassifiedImage();
    //cv::waitKey();
    
    //cout << "firstAlg_img done" << endl;
    
    cout << "Comparing GT to algorithm " << firstAlgorithm << endl;
    for  (int i=1; i<class_coordinates.size() ; i++)    // for each class
    {
        if (class_coordinates[i].size() < 1) {
            continue;
        }
        for (int k = 0; k < class_coordinates[i].size(); k++){
            Point tempPt = class_coordinates[i][k];
            int gtClass = i;    // assuming also comparing unknowns(0)
            Vec3b hyperfuncClass = HyperFunctions1.classified_img.at<Vec3b>(tempPt.x,tempPt.y);
            //cout<<hyperfuncClass<<endl;
            if (gtClass == hyperfuncClass[0]) {
                accuracy_by_class[i] += 1;
            }
        }
        cout << "Accuracy of class:   " << i<<"  "<<class_list[i] << ":  " << (double)(accuracy_by_class[i] ) <<" / "<<class_coordinates[i].size()<< "   "<<(double)(accuracy_by_class[i] ) /(double)class_coordinates[i].size() *100 <<"%"<< endl;
    }
    
 


    // setting firstAlg as ground "ground truth"
    Mat firstAlg_img_Classified, firstAlg_img_Classified_normal;
    firstAlg_img_Classified = HyperFunctions1.classified_img;

    // visualize results of firstAlg and gt img
    normalize(firstAlg_img_Classified, firstAlg_img_Classified_normal, 0,255, NORM_MINMAX, CV_8UC1);
    imshow("firstAlg img", firstAlg_img_Classified_normal);
    Mat gt_normal;
    normalize(gt_img, gt_normal, 0,255, NORM_MINMAX, CV_8UC1);
    imshow("gt img", gt_normal);
    //cv::waitKey();




    // Sets it to secondAlg for comparison
    HyperFunctions1.spec_sim_alg = secondAlgorithm; 
    HyperFunctions1.SemanticSegmenter();

    //For each pixel in the secondAlg classified image, compare to the firstAlg one
    vector<double> accuracy_by_class2 (class_coordinates.size());
    vector<double> firstAlg_in_class (class_coordinates.size()); //How many pixels the firstAlg said were in the class
    vector<double> secondAlg_in_class (class_coordinates.size()); //How many pixels the secondAlg said were in the class
    cout << "Comparing algorithms " << firstAlgorithm << " and " << secondAlgorithm << endl;
    for  (int i=0; i<HyperFunctions1.classified_img.rows ; i++)
    {
        for (int k = 0; k < HyperFunctions1.classified_img.cols; k++){
            Vec3b firstAlg_Class = firstAlg_img_Classified.at<Vec3b>(i,k);    
            Vec3b secondAlg_Class = HyperFunctions1.classified_img.at<Vec3b>(i,k);
            if (firstAlg_Class == secondAlg_Class) {
                accuracy_by_class2[firstAlg_Class[0]] += 1;
            }
            firstAlg_in_class[firstAlg_Class[0]] += 1;
            secondAlg_in_class[secondAlg_Class[0]] += 1;
        }
        
    }

    
    for (int i = 1; i < class_coordinates_size; i++) {
        if (class_coordinates[i].size() < 1) {
            continue;
        }
        cout << "secondAlg percentage agreement with firstAlg class " << i << ": " << 100*(double)(accuracy_by_class2[i] ) / firstAlg_in_class[i]<< "%" <<endl;
        cout << "firstAlg percentage agreement with secondAlg class " << i << ": " << 100*(double)(accuracy_by_class2[i] ) / secondAlg_in_class[i]<< "%" <<endl;
    }


    Mat secondAlg_img_Classified, secondAlg_img_Classified_normal;
    secondAlg_img_Classified = HyperFunctions1.classified_img;
    normalize(secondAlg_img_Classified, secondAlg_img_Classified_normal, 0,255, NORM_MINMAX, CV_8UC1);
    imshow("secondAlg img", secondAlg_img_Classified_normal);
    cv::waitKey();
  
  return 0;
}
