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
    
    //ENVI is stored as a pair of DAT and HDR
    const char* dat_file = "../../HyperImages/test.dat";

    //Ground truth png
    string gt_file = "../../HyperImages/labeledtest.png";
    
    //name of spectral database being created
    string spectral_database="../json/envi_spectral_database.json";
    
    // Register GDAL drivers
    GDALAllRegister();

    //const char* enviHeaderPath = "../../HyperImages/test.dat";

    std::vector<cv::Mat> imageBands;

    // Open the ENVI file
    GDALDataset *poDataset = (GDALDataset *) GDALOpen(dat_file, GA_ReadOnly);
    
    if (poDataset != nullptr) {
        // Get information about the dataset
        int width = poDataset->GetRasterXSize();
        int height = poDataset->GetRasterYSize();
        int numBands = poDataset->GetRasterCount();

        printf("Width: %d, Height: %d, Bands: %d\n", width, height, numBands);

        // Loop through bands and read data
        for (int bandNum = 1; bandNum <= 5/*numBands*/; ++bandNum) {
            GDALRasterBand *poBand = poDataset->GetRasterBand(bandNum);

            // Allocate memory to store pixel values
            int *bandData = (int *) CPLMalloc(sizeof(int) * width * height);

            // Read band data
            poBand->RasterIO(GF_Read, 0, 0, width, height, bandData, width, height, GDT_Int32, 0, 0);

            // Create an OpenCV Mat from the band data
            cv::Mat bandMat(height, width, CV_32SC1, bandData);

            // Add the Mat to the vector
            imageBands.push_back(bandMat);

            // Release allocated memory
            CPLFree(bandData);
        }

        // Close the dataset
        GDALClose(poDataset);

    } else {
        printf("Failed to open the dataset.\n");
    }

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
    for (const auto& item : items) {
        string name = item["name"].asString();
        string colorHexCode = item["color_hex_code"].asString();

        class_list.push_back(name);
        colorIndexVector.push_back({colorHexCode, i});
        i++; //Rewriting this loop to look better might be a better idea
    }

    // Print the results
    cout << "Class List:" << endl;
    for (const auto& className : class_list) {
        cout << className << endl;
    }

    cout << "\nColor Index Vector:" << endl;
    for (const auto& pair : colorIndexVector) {
        cout << "Color: " << pair.first << ", Index: " << pair.second << endl;
    }

    HyperFunctions HyperFunctions1;
    // load hyperspectral image
    HyperFunctions1.mlt1 = imageBands;
    //HyperFunctions1.LoadImageHyper(file_name1);

    // load ground truth image
    Mat gt_img = imread(gt_file, IMREAD_COLOR);
    
    // make sure ground truth image is 8 bit single channel
    // ref https://gist.github.com/yangcha/38f2fa630e223a8546f9b48ebbb3e61a
    //cout<<gt_img.type()<<endl;
    /*
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
    */
    
    // get number of semantic classes 
    // assumption 0 is unknown
    // class values 1-N (N is total number of classes)
    // only care about max value
    /*
    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;
    minMaxLoc( gt_img, &minVal, &maxVal, &minLoc, &maxLoc );
    cout<<"Number of semantic classes: "<<maxVal<<endl;
    
    if (maxVal<1)
    {
        cout<<"improper input"<<endl;
        return -1;
    }*/

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
                which_class = it->second;
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

    // verify number of samples is right
    /*cout << "class and number of samples in class" << endl;
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
                avgSpectrums[i][j] += HyperFunctions1.mlt1[j].at<uchar>(tempPt);
            }
        }

        for  (int j=0; j<HyperFunctions1.mlt1.size() ; j++)
        {
            avgSpectrums[i][j] /= class_coordinates[i].size();
            if (avgSpectrums[i][j]<0){ avgSpectrums[i][j]=0;}
            if (avgSpectrums[i][j]>255){ avgSpectrums[i][j]=255;}
            avgSpectrums_vector[i][j]=avgSpectrums[i][j];
        }
    }


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
            if (avgSpectrums[j][i]<0){ avgSpectrums[j][i]=0;}
            if (avgSpectrums[j][i]>255){ avgSpectrums[j][i]=255;}
            value_obj["Spectral_Information"][class_list[j]][zero_pad_result] = avgSpectrums_vector[j][i]; //value between 0-255 corresponding to reflectance

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
    
    cout<<"spectral database was saved to json file"<<endl;

    // read the json file that was just written so everything is in correct data structure
    HyperFunctions1.read_ref_spec_json(spectral_database);
    
    // show spectral similarity image 
    /*
    HyperFunctions1.spec_sim_alg = 0;
    HyperFunctions1.SpecSimilParent();
    HyperFunctions1.DispSpecSim();
    cv::waitKey();*/
    

    // set alg to SAM and perform semantic segmentation
    //cout << "Segmenting Image with SAM algorithm" << endl;
    HyperFunctions1.spec_sim_alg = 0;
    auto start = high_resolution_clock::now();
    HyperFunctions1.SemanticSegmenter();
    auto end = high_resolution_clock::now();
     cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
    //HyperFunctions1.DispClassifiedImage();
    //cv::waitKey();
    
    //cout << "SAM_img done" << endl;
    
    cout << "Comparing GT to SAM algorithm" << endl;
    for  (int i=1; i<class_coordinates.size() ; i++)    // for each class
    {
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
    
 


    // setting SAM as ground "ground truth"
    Mat SAM_img_Classified, SAM_img_Classified_normal;
    SAM_img_Classified = HyperFunctions1.classified_img;

    // visualize results of sam and gt img
    normalize(SAM_img_Classified, SAM_img_Classified_normal, 0,255, NORM_MINMAX, CV_8UC1);
    imshow("sam img", SAM_img_Classified_normal);
    Mat gt_normal;
    normalize(gt_img, gt_normal, 0,255, NORM_MINMAX, CV_8UC1);
    imshow("gt img", gt_normal);
    //cv::waitKey();




    // Sets it to SCM for comparison
    HyperFunctions1.spec_sim_alg = 1; 
    HyperFunctions1.SemanticSegmenter();

    //For each pixel in the SCM classified image, compare to the SAM one
    vector<double> accuracy_by_class2 (class_coordinates.size());
    vector<double> SAM_in_class (class_coordinates.size()); //How many pixels the SAM said were in the class
    vector<double> SCM_in_class (class_coordinates.size()); //How many pixels the SAM said were in the class
    cout << "Comparing SAM to SCM results" << endl;
    for  (int i=0; i<HyperFunctions1.classified_img.rows ; i++)
    {
        for (int k = 0; k < HyperFunctions1.classified_img.cols; k++){
            Vec3b SAM_Class = SAM_img_Classified.at<Vec3b>(i,k);    
            Vec3b SCM_Class = HyperFunctions1.classified_img.at<Vec3b>(i,k);
            if (SAM_Class == SCM_Class) {
                accuracy_by_class2[SAM_Class[0]] += 1;
            }
            SAM_in_class[SAM_Class[0]] += 1;
            SCM_in_class[SCM_Class[0]] += 1;
        }
        
    }

    
    for (int i = 1; i < class_coordinates_size; i++) {
        cout << "SCM percentage agreement with SAM class " << i << ": " << 100*(double)(accuracy_by_class2[i] ) / SAM_in_class[i]<< "%" <<endl;
        cout << "SAM percentage agreement with SCM class " << i << ": " << 100*(double)(accuracy_by_class2[i] ) / SCM_in_class[i]<< "%" <<endl;
    }


    Mat SCM_img_Classified, SCM_img_Classified_normal;
    SCM_img_Classified = HyperFunctions1.classified_img;
    normalize(SCM_img_Classified, SCM_img_Classified_normal, 0,255, NORM_MINMAX, CV_8UC1);
    imshow("scm img", SAM_img_Classified_normal);
    cv::waitKey();
  
  return 0;
}
