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
#include <filesystem>

#if use_cuda 
#include "../src/hypergpufunctions.cu"

#endif


using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace Json;

int main (int argc, char *argv[])
{



    // parent folder to analyze
    string lib_hsi_dir = "../../HyperImages/LIB-HSI/LIB-HSI/validation/";

    // extensions to search for hyperspectral image
    string envi_ext = ".dat";

    // extension to search for ground truth image
    string gt_ext = ".png";

    // database with gt info about classes 
    string gt_database="../json/lib_hsi.json";

    // used to find average spectrum for each semantic class, not needed if already processed
    bool get_average_spectrum = true;

    // directory for results with statistics 
    string results_dir = lib_hsi_dir + "results/";
    
    // directory for spectral database for each image 
    string spec_database_dir = lib_hsi_dir + "spectral_databases/";



    HyperFunctions HyperFunctions1;
    vector<string> envi_files;
    vector<string> gt_files;
    vector<string> spectral_database_files;
    int k=0;

    //assume that there is a gt file for each envi file

    // get all the hyperspectral envi files and gt labeled files
    for (auto &p : std::filesystem::recursive_directory_iterator(lib_hsi_dir))
    {
        if (p.path().extension() == envi_ext)
        {
            
            k++;
            string temp_string=p.path().stem().string();
            // cout<<temp_string <<endl;
            string temp_path_envi = p.path().string();//lib_hsi_dir + "reflectance_cubes/"+temp_string + envi_ext;
            // location of corresponding ground truth image
            string temp_path_gt = lib_hsi_dir + "labels/"+temp_string + gt_ext;
            // cout<<temp_path_envi<<" "<<temp_path_gt<<endl;
            envi_files.push_back(temp_path_envi);
            gt_files.push_back(temp_path_gt);
        }
            
    }

    cout<<"number of images to be analyzed: "<<k<<endl;


    // Read the JSON file with gt info about rgb images
    ifstream ifs(gt_database);
    if (!ifs.is_open()) {
        cerr << "Error opening JSON file." << endl;
        return -1;
    }
    
    
    // Parse the JSON file with color and class information
    CharReaderBuilder readerBuilder;
    Value jsonData;
    parseFromStream(readerBuilder, ifs, &jsonData, nullptr);

    // Extract class names and color hex codes
    vector<string> class_list;
    vector<string> colorHexVector;
    vector<Vec3b> colorBGRVector;

    const Value& items = jsonData["items"];
    int i = 0;
    // class_list.push_back("unknown");
    for (const auto& item : items) {
        string name = item["name"].asString();
        string colorHexCode = item["color_hex_code"].asString();

        class_list.push_back(name);
        colorHexVector.push_back(colorHexCode);
        i++; //Rewriting this loop to look better might be a better idea
        // cout<<name<<" "<<colorHexCode<<endl;
    }

    
    // convert hex colors to rgb
    i=0;
    for (const auto& hex : colorHexVector) {
        
        int r = std::stoi(hex.substr(1, 2), nullptr, 16);
        int g = std::stoi(hex.substr(3, 2), nullptr, 16);
        int b = std::stoi(hex.substr(5, 2), nullptr, 16);
        cv::Vec3b bgr_value;
        bgr_value = cv::Vec3b(b, g, r);
        // cout << class_list[i] <<"  Hex: " << hex << ", BGR: " << bgr_value << endl;
        colorBGRVector.push_back(bgr_value);
        i++;
    }

    Mat gt_img;
    Vec3b temp_val;
    bool all_pixel_values_valid = true;
    int numClasses = colorHexVector.size(); 
    cout<<"number of semantic classes: "<<numClasses<<endl;

    //make sure pixel values are all valid by going through each pixel in each gt image
    // int total_num_pixels = 0;
    for (int i=0; i<gt_files.size(); i++)
    {
        gt_img = imread(gt_files[i], IMREAD_COLOR);
        for  (int i=0; i<gt_img.rows ; i++)
        {
            for (int j=0; j<gt_img.cols ; j++)
            {
                temp_val=gt_img.at<Vec3b>(i,j);
                // total_num_pixels++;
                //make sure pixel rgb values are valid
                auto it = std::find(colorBGRVector.begin(), colorBGRVector.end(), temp_val);

                if (it != colorBGRVector.end()) {
                    // temp_val is in colorBGRVector
                } else {
                    // temp_val is not in colorBGRVector
                    cout<<"error: pixel value not in colorBGRVector "<< temp_val<<endl;
                    all_pixel_values_valid = false;
                }

            }
        }
    }
    // cout<<"total number of pixels: "<<total_num_pixels<<endl;

    // stop if there are invalid pixels
    if (!all_pixel_values_valid)
    {
        cout<<"error: pixel values not valid"<<endl;
        return -1;
    }
    else
    {
        cout<<"all pixel values are valid"<<endl;
    }


    

    // see what mat type ground truth images are in for analysis
    //  Mat test_img = imread(gt_files[0], IMREAD_COLOR);
    //  cout<<test_img.type()<<endl; // type 16, cv_8uc3
    //  cout<<test_img.at<Vec3b>(10,10)<<endl;



    // // test visualization of rgb of hyperspectral image and gt image
    // for (int i=0; i<envi_files.size(); i++)
    // {
    //     imshow("gt", imread(gt_files[i], IMREAD_COLOR));
    //     HyperFunctions1.LoadImageHyper(envi_files[i]);
    //     HyperFunctions1.false_img_b = HyperFunctions1.mlt1.size()/3;
    //     HyperFunctions1.false_img_g = HyperFunctions1.mlt1.size()*2/3;
    //     HyperFunctions1.false_img_r = HyperFunctions1.mlt1.size()-1;
    //     HyperFunctions1.GenerateFalseImg();
    //     imshow("false img", HyperFunctions1.false_img);
    //     waitKey(300);
    // }


    
    
    // make a directory for results
    std::filesystem::create_directory(results_dir);
    std::filesystem::create_directory(spec_database_dir);
    
    
    // go through each image and create spectral database
    if (get_average_spectrum)
    {
        
        for (int i=0; i<gt_files.size(); i++)
        // for (int i=0; i<3; i++)
        {
            // load ground truth image
            gt_img = imread(gt_files[i], IMREAD_COLOR);
            // // load envi hyperspectral image
            HyperFunctions1.LoadImageHyper(envi_files[i]);

            // initialize json file to save average spectrum for each class
            // save spectral database to json file
            string spectral_database_name = spec_database_dir + std::filesystem::path(envi_files[i]).stem().string() + ".json";
            // cout<<spectral_database_name<<endl;
            std::ofstream file_id;
            file_id.open(spectral_database_name);
            Json::Value value_obj;



            struct Vec3bCompare {
                bool operator() (const cv::Vec3b &a, const cv::Vec3b &b) const {
                    return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
                }
            };

            std::map<cv::Vec3b, std::vector<cv::Point>, Vec3bCompare> colorCoordinates;

            for (int y = 0; y < gt_img.rows; y++) {
                for (int x = 0; x < gt_img.cols; x++) {
                    cv::Vec3b color = gt_img.at<cv::Vec3b>(cv::Point(x, y));
                    colorCoordinates[color].push_back(cv::Point(x, y));
                }
            }
           
            // cout<<"number of classes: "<<colorCoordinates.size()<<endl;
            
            // print out all the colors in the image and the number of pixels in each class and the rgb values
            for (auto const& x : colorCoordinates)
            {
                auto it = std::find(colorBGRVector.begin(), colorBGRVector.end(), x.first);

                if (it != colorBGRVector.end()) {
                    // std::cout << "Color found at index " << std::distance(colorBGRVector.begin(), it) << " name: "<< class_list[ std::distance(colorBGRVector.begin(), it)]<< " " <<x.first << " " << x.second.size() << endl;
                } else {
                    std::cout << "Color not found" << std::endl;
                }

                // get the average spectrum for each class and save to a json file

                vector<unsigned  long int> average_spectrum (HyperFunctions1.mlt1.size(), 0);
                // go through each pixel in the class and add to average spectrum then divide by number of pixels
                for (int j=0; j<x.second.size(); j++)
                {
                    // cout<<x.second[j].x<<" "<<x.second[j].y<<endl;
                    for (int k=0; k<HyperFunctions1.mlt1.size(); k++)
                    {
                        average_spectrum[k] += HyperFunctions1.mlt1[k].at<uchar>(x.second[j].y, x.second[j].x);
                    }
                }
                for (int k=0; k<HyperFunctions1.mlt1.size(); k++)
                {
                    average_spectrum[k] /= x.second.size(); 
                }


                
                // //cout average_spectrum
                // for (int i=0; i<average_spectrum.size(); i++)
                // {
                //     cout<<average_spectrum[i]<<" ";
                // }
                // cout<<endl;

                // save average spectrum to json file
                value_obj["Color_Information"][class_list[ std::distance(colorBGRVector.begin(), it)]]["red_value"] = colorBGRVector[std::distance(colorBGRVector.begin(), it)][0];
                value_obj["Color_Information"][class_list[ std::distance(colorBGRVector.begin(), it)]]["green_value"] = colorBGRVector[std::distance(colorBGRVector.begin(), it)][2];
                value_obj["Color_Information"][class_list[ std::distance(colorBGRVector.begin(), it)]]["blue_value"] = colorBGRVector[std::distance(colorBGRVector.begin(), it)][1];



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
                    if ( average_spectrum[i]<0){ average_spectrum[i]=0;}
                    if ( average_spectrum[i]>255){  average_spectrum[i]=255;}
                    value_obj["Spectral_Information"][class_list[ std::distance(colorBGRVector.begin(), it)]][zero_pad_result] = static_cast<int>(average_spectrum[i]); //value between 0-255 corresponding to reflectance
                    
                }
            
            

            }
            
            Json::StyledWriter styledWriter;
            file_id << styledWriter.write(value_obj);
            file_id.close();
            cout<< i<< " spectral database was saved to json file: " << spectral_database_name << endl;
            spectral_database_files.push_back(spectral_database_name);
           
            // return -1;
            
            
            // cout<<colorCoordinates[colorBGRVector[0]][0]<<endl;

            // imshow("gt", gt_img);
            // // cv::waitKey();

            
            // HyperFunctions1.false_img_b = HyperFunctions1.mlt1.size()/3;
            // HyperFunctions1.false_img_g = HyperFunctions1.mlt1.size()*2/3;
            // HyperFunctions1.false_img_r = HyperFunctions1.mlt1.size()-1;
            // HyperFunctions1.GenerateFalseImg();
            // HyperFunctions1.DispFalseImage();
            // cv::waitKey();
            

        } // end of going through each image
    } //end if get_average_spectrum




    // go through hyperspectral images and perform semantic segmentation for each algorithm
    for (int img_index = 0; img_index < envi_files.size(); img_index++)
    {
        HyperFunctions1.LoadImageHyper(envi_files[img_index]);
        string spec_data_name = spec_database_dir + std::filesystem::path(envi_files[img_index]).stem().string() + ".json";
        HyperFunctions1.read_ref_spec_json(spec_data_name);
        // HyperFunctions1.read_spectral_json(spec_data_name);
        Mat gt_img2 = imread(gt_files[img_index], IMREAD_COLOR);
        // cout<<"hyperspectral image: "<<envi_files[img_index]<<"  gt image: "<< gt_files[img_index]<<endl;
        
        // print results to json file  so initialize here
        std::ofstream file_id2;
        string result_database_name = results_dir + std::filesystem::path(envi_files[img_index]).stem().string() + ".json";
        file_id2.open(result_database_name);
        Json::Value value_obj2;

        // will need to iterate over all the algorithms  with a loop 
        HyperFunctions1.spec_sim_alg = 0;
        int total_sum_pixels=0;
        vector<Vec3b> found_colors;

        // algorithms are from 0-14
        // change below for full test
        for (int spec_sim_val=0; spec_sim_val<15; spec_sim_val++)
        // for (int spec_sim_val=0; spec_sim_val<2; spec_sim_val++)
        {
            HyperFunctions1.spec_sim_alg = spec_sim_val;

            auto start = high_resolution_clock::now();
            HyperFunctions1.SemanticSegmenter();
            auto end = high_resolution_clock::now();
            // cout << "Time taken : " << (float)duration_cast<milliseconds>(end-start).count() / (float)1000 << " " << "seconds"<<endl;
            
            // make sure images are same size 
            if (HyperFunctions1.classified_img.rows != gt_img2.rows || HyperFunctions1.classified_img.cols != gt_img2.cols)
            {
                cout<<"error: classified_img and gt_img2 are not same size"<<endl;
                return -1;
            }



            // check if all pixels are valid
            for  (int i=0; i<HyperFunctions1.classified_img.rows ; i++)
            {
                for (int j=0; j<HyperFunctions1.classified_img.cols ; j++)
                {
                    temp_val=HyperFunctions1.classified_img.at<Vec3b>(i,j);

                    //make sure pixel rgb values are valid
                    auto it = std::find(colorBGRVector.begin(), colorBGRVector.end(), temp_val);

                    if (it != colorBGRVector.end()) {
                        // temp_val is in colorBGRVector
                    } else {
                        // temp_val is not in colorBGRVector
                        cout<<"error: pixel value not in colorBGRVector "<< temp_val<<endl;
                        all_pixel_values_valid = false;
                        return -1;
                    }

                }
            }


            // compare classified image to ground truth image 
            // find true positives, false positives, false negatives, true negatives

            int true_positives = 0, false_positives = 0, false_negatives = 0, true_negatives = 0;
            for (int i=0; i<HyperFunctions1.classified_img.rows; i++)
            {
                for (int j=0; j<HyperFunctions1.classified_img.cols; j++)
                {
                    // cout<<HyperFunctions1.classified_img.at<Vec3b>(i,j)<<endl;
                    // cout<<gt_img2.at<Vec3b>(i,j)<<endl;
                    if (HyperFunctions1.classified_img.at<Vec3b>(i,j) == gt_img2.at<Vec3b>(i,j))
                    {
                        true_positives++;
                    }
                    else
                    {
                        // is this right?
                        false_positives++;
                    }
                }
            }


            value_obj2["Spectral Similarity Algorithm"][to_string(HyperFunctions1.spec_sim_alg)]["Time"] = (float)duration_cast<milliseconds>(end-start).count() / (float)1000;
            // value_obj2["Spectral Similarity Algorithm"][HyperFunctions1.spec_sim_alg]["Number of Classes"] = HyperFunctions1.reference_colors.size();
            value_obj2["Spectral Similarity Algorithm"][to_string(HyperFunctions1.spec_sim_alg)]["Number of Classes"] = static_cast<unsigned int>(HyperFunctions1.color_combos.size());

            value_obj2["Spectral Similarity Algorithm"][to_string(HyperFunctions1.spec_sim_alg)]["True Positive"] = true_positives;

            value_obj2["Spectral Similarity Algorithm"][to_string(HyperFunctions1.spec_sim_alg)]["False Positive"] = false_positives;
            value_obj2["Spectral Similarity Algorithm"][to_string(HyperFunctions1.spec_sim_alg)]["Number of Pixels"] = true_positives + false_positives ;
                


            // display gt and classified image
            // imshow("gt", gt_img2);
            // imshow("classified", HyperFunctions1.classified_img);
            // cv::waitKey();
        
        } // end spec sim loop

        // cout<<"total sum pixels: "<<total_sum_pixels<<endl;
        
       




        Json::StyledWriter styledWriter2;
        file_id2 << styledWriter2.write(value_obj2);
        file_id2.close();

        cout<< img_index << " result database was saved to json file: " << result_database_name << endl;


        // below is to visualize the results
        // imshow("gt", imread(gt_files[img_index], IMREAD_COLOR));
        // HyperFunctions1.false_img_b = HyperFunctions1.mlt1.size()/3;
        // HyperFunctions1.false_img_g = HyperFunctions1.mlt1.size()*2/3;
        // HyperFunctions1.false_img_r = HyperFunctions1.mlt1.size()-1;
        // HyperFunctions1.GenerateFalseImg();
        // HyperFunctions1.DispFalseImage();
        // HyperFunctions1.DispClassifiedImage();
        // cv::waitKey();



        // return -2;

    } // end envi analysis loop

  return 0;
}
