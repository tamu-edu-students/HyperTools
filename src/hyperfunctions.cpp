#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "hyperfunctions.h"
#include <fstream>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <stdio.h>
#include "ctpl.h"
#include "opencv2/xfeatures2d.hpp"
#include "spectralsimalgorithms.cpp"
#include "gdal/gdal.h"
#include "gdal/gdal_priv.h"
#include "gdal/cpl_conv.h"
#include <matio.h>  

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// Loads first hyperspectral image for analysis
void HyperFunctions::LoadImageHyper(string file_name, bool isImage1 = true)
{
    // if (isImage1) {
    //     mlt1.clear();
	//     imreadmulti(file_name, mlt1);
    // }
    // else {
    //     mlt2.clear();
	//     imreadmulti(file_name, mlt2);
    // }


    if (isImage1) {
        mlt1.clear();
    }
    else
    {
        mlt2.clear();
    }
	string file_ext;

    size_t dotPos = file_name.find_last_of('.');
    if (dotPos != std::string::npos) {
        file_ext = file_name.substr(dotPos + 1);
    }

    if (file_ext=="tiff" || file_ext=="tiff")
    {
        if (isImage1) {
            imreadmulti(file_name, mlt1);
        }
        else {
            imreadmulti(file_name, mlt2);
        }
    }
    else if (file_ext=="dat"||file_ext=="hdr")
    {

        // right now this is for dat files 
        // assumes data type = 4

        // Register GDAL drivers
        GDALAllRegister();

        const char* enviHeaderPath = const_cast<char*>( file_name.c_str());
        GDALDataset *poDataset = (GDALDataset *) GDALOpen(enviHeaderPath, GA_ReadOnly);
        if (poDataset != nullptr) {
            // Get information about the dataset
            int width = poDataset->GetRasterXSize();
            int height = poDataset->GetRasterYSize();
            int numBands = poDataset->GetRasterCount();

            // printf("Width: %d, Height: %d, Bands: %d\n", width, height, numBands);

            std::vector<cv::Mat> imageBands;
            
            // Loop through bands and read data
            for (int bandNum = 1; bandNum <= numBands; ++bandNum) {
                GDALRasterBand *poBand = poDataset->GetRasterBand(bandNum);

                // Allocate memory to store pixel values
                float *bandData = (float *) CPLMalloc(sizeof(float) * width * height);

                // Read band data
                CPLErr err =  poBand->RasterIO(GF_Read, 0, 0, width, height, bandData, width, height, GDT_Float32, 0, 0);

                if (err != CE_None) {
                    // Handle the error
                    std::cerr << "Error reading band data: " << CPLGetLastErrorMsg() << std::endl;
                }

                // Create an OpenCV Mat from the band data
                cv::Mat bandMat(height, width, CV_32FC1, bandData);

                // corrects the orientation of the image
                bandMat = bandMat.t(); 
                cv::flip(bandMat, bandMat, 1); 

                
                // problem stems from improper calibration if there are areas of image that are dark that shouldnt be
                cv::threshold(bandMat, bandMat, 1.0, 1.0, cv::THRESH_TRUNC);

                bandMat.convertTo(bandMat, CV_8UC1, 255.0);

                // below is for visualization
                //cv::imshow("bandMat", bandMat);
                //cv::waitKey(30);


                if (isImage1) {
                    mlt1.push_back(bandMat);
                }
                else {
                    mlt2.push_back(bandMat);
                }

                // Release allocated memory
                CPLFree(bandData);
            }

            // Close the dataset
            GDALClose(poDataset);

        } else {
            printf("Failed to open the dataset.\n");
        }


    }
    else if (file_ext=="mat")
    {
        // Open the MATLAB file
        mat_t *matfp = Mat_Open(file_name.c_str(), MAT_ACC_RDONLY);
        if (matfp == NULL) {
            std::cerr << "Error opening MATLAB file " << file_name << std::endl;
            return ;
        }

        // Read each variable in the file
        matvar_t *matvar = NULL;
        while ((matvar = Mat_VarReadNext(matfp)) != NULL) {


            // cout << "Variable type: ";
            // switch (matvar->class_type) {
            //     case MAT_C_DOUBLE: cout << "double"; break;
            //     case MAT_C_SINGLE: cout << "single"; break;
            //     case MAT_C_INT32: cout << "int32"; break;
            //     case MAT_C_UINT8: cout << "uint8"; break;
            //     // Add more cases as needed...
            //     default: cout << "unknown";
            // }
            // cout << endl;

            // hyperspectral image is of type double
            
            // cout << "Variable dimensions: ";
            // for (int i = 0; i < matvar->rank; i++) {
            //     cout << matvar->dims[i];
            //     if (i < matvar->rank - 1) {
            //         cout << " x ";
            //     }
            // }
            // cout << endl;

            // Assuming matvar is your 3D array variable
            // x,y,chanel
            if (matvar->rank == 3) {
                size_t rows = matvar->dims[0];
                size_t cols = matvar->dims[1];
                size_t slices = matvar->dims[2];

                double* data = static_cast<double*>(matvar->data);

                // Find global minimum and maximum
                double global_min = *std::min_element(data, data + rows * cols * slices);
                double global_max = *std::max_element(data, data + rows * cols * slices);


                for (size_t i = 0; i < slices; i++) {
                    cv::Mat mat(rows, cols, CV_64F, data + i * rows * cols);
                    // normalize the data
                    mat = (mat - global_min) * 255 / (global_max - global_min);
                    // convert to 8 bit
                    mat.convertTo(mat, CV_8U);
                    // cv::imshow("mat", mat);
                    // cv::waitKey(100);
                    // cout<<i<<endl;
                    if (isImage1) {
                        mlt1.push_back(mat);
                    }
                    else {
                        mlt2.push_back(mat);
                    }
                }
            }

            // Free the current MATLAB variable
            Mat_VarFree(matvar);
        }

        // Close the MATLAB file
        Mat_Close(matfp);
    }
    else
    {
        cout<<"file extension not supported"<<endl;
    }
}

// Loads a segmented or classified image
// mainly used to reprocess classified images through filtering and polygon simplification
void HyperFunctions::LoadImageClassified(string file_name)
{
    classified_img = cv::imread(file_name);
}

// loads the first grayscale image for feature analysis
void HyperFunctions::LoadFeatureImage1(string file_name)
{
    feature_img1 = cv::imread(file_name, IMREAD_GRAYSCALE);
}

//  loads the second grayscale image for feature analysis
void HyperFunctions::LoadFeatureImage2(string file_name)
{
    feature_img2 = cv::imread(file_name, IMREAD_GRAYSCALE);
}

// Displays side by side feature images
void HyperFunctions::DispFeatureImgs()
{
    Mat temp_img, temp_img2, temp_img3;
    cv::resize(feature_img1, temp_img2, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    cv::resize(feature_img2, temp_img3, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    Mat matArray1[] = {temp_img2, temp_img3};
    hconcat(matArray1, 2, temp_img);
    cv::resize(temp_img, temp_img, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    feature_img_combined = temp_img;
    //    imshow("Feature Images ", feature_img_combined);
}

// GA-ORB turning hyperspectral into 2-D

void HyperFunctions::gaSpace(bool isImage1)
{
    // assumes mlt1 and mlt2 have same spatial and spectral resolution
    Mat output_image(mlt1[0].rows, mlt1[0].cols, CV_16U, cv::Scalar(0));
    int numChannels = mlt1.size();

    int sumTot = 0;
    int temp_val2;
    for (int i = 0; i < mlt1[0].rows; i++)
    {
        for (int k = 0; k < mlt1[1].cols; k++)
        {
            for (int n = 0; n < numChannels; n++)
            {

                if (isImage1)
                {
                    temp_val2 = mlt1[n].at<uchar>(i, k);
                }
                else
                {
                    temp_val2 = mlt2[n].at<uchar>(i, k);
                }
                sumTot += temp_val2;
            }

            output_image.at<ushort>(i, k) = sumTot;
            sumTot = 0;
        }
    }

    ga_img = output_image;
    // convert to Mat data type that is compatible with Fast
    normalize(ga_img, ga_img, 0, 255, NORM_MINMAX, CV_8U);
    // imshow("Output Image", output_image);
    // cv::waitKey();
    // return output_image;
}
// Integral Image
void HyperFunctions::ImgIntegration()
{
    // Checks if image is empty
    if (ga_img.empty())
    {
        std::cerr << "Error: Input image is empty." << std::endl;
    }
    // Computes image integration
    //  computing a integral image based off of ga_img
    cv::integral(ga_img, integral_img, CV_32F);
    normalize(integral_img, integral_img, 0, 255, NORM_MINMAX, CV_8U);
}

void HyperFunctions::CreateCustomFeatureDetector(int hessVal, vector<KeyPoint> &keypoints, Mat feature_img)
{
    for (int y = 0; y < feature_img.rows; y += hessVal)
    {
        for (int x = 0; x < feature_img.cols; x += hessVal)
        {
            keypoints.push_back(cv::KeyPoint(static_cast<float>(x), static_cast<float>(y), 1));
        }
    }

    drawKeypoints(feature_img, keypoints, feature_img);
}

void HyperFunctions::SSDetector(const cv::Mat &hyperspectralCube, std::vector<cv::KeyPoint> &keypoints)
{
    // const float M_max = 1.0;
    // cv::Mat hyperspectralCube; // we need a hyperspectralcube data

    // final keypoints vector

    double sigma1 = 1.6; // variables are set based on what the paper said.
    double sigma2 = 1.8;

    cv::GaussianBlur(hyperspectralCube, hyperspectralCube, cv::Size(0, 0), sigma1, sigma2, cv::BORDER_DEFAULT);

    int octaveLevels = 3;
    double k = 2.0;
    cv::Mat previousScale;
    cv::GaussianBlur(hyperspectralCube, previousScale, cv::Size(0, 0), sigma1, sigma2, cv::BORDER_DEFAULT);
    for (int octave = 1; octave <= octaveLevels; ++octave)
    {

        cv::Mat currentScale;
        cv::GaussianBlur(hyperspectralCube, currentScale, cv::Size(0, 0), sigma1, sigma2, cv::BORDER_DEFAULT);
        cv::Mat dog = currentScale - previousScale;

        // std::vector<cv::KeyPoint> currentKeypoints;

        // cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
        cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();

        detector->detect(dog, keypoints);

        // filtering the keypoints

        // keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint &keypoints) {
        // return keypoints.response <= 0.75;
        // }), keypoints.end());

        /*keypoints1.erase(std::remove_if(keypoints2.begin(), keypoints2.end(), [](const cv::KeyPoint &keypoint) {
         return keypoint.response <= 0.75;
         }), keypoints.end());*/

        // updating for the next octave
        previousScale = currentScale.clone();

        sigma1 *= k;
        sigma2 *= k;
    }
    // for (const auto& kp : keypoints)
    //       {
    //      std::cout << "x_1: " << kp.pt.x << ", y_1: " << kp.pt.y << std::endl;
    //      }
}



void HyperFunctions::SSDescriptors(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, cv::Mat &descriptor1, cv::Mat &descriptor2, float M_max = 1.0)
{
    const int numThetaBins = 8;
    const int numPhiBins = 4;
    const int numGradientBins = 8;
    const int descriptorSize = numThetaBins * numPhiBins * numGradientBins;

    float M = 1.0;

    // Process keypoints1
    descriptor1 = cv::Mat::zeros(keypoints1.size(), descriptorSize, CV_32F);
    for (size_t i = 0; i < keypoints1.size(); ++i)
    {
        cv::Mat descriptor_1 = descriptor1.row(i);
        const cv::KeyPoint& kp = keypoints1[i];

        for (int x = -8; x <= 7; ++x)
        {
            for (int y = -8; y <= 7; ++y)
            {
                float theta = std::atan2(kp.pt.y -  y, kp.pt.x - x) * (180.0 / CV_PI);
                float phi = 0.0; // Since there is no z component

                int thetaBin = static_cast<int>(theta / (360.0 / numThetaBins));
                int phiBin = static_cast<int>((phi + 90.0) / (180.0 / numPhiBins));
                int gradientBin = static_cast<int>(M / (M_max / numGradientBins));

                int index = thetaBin * numPhiBins * numGradientBins + phiBin * numGradientBins + gradientBin;
                descriptor_1.at<float>(0, index) += M;
            }
        }

        cv::normalize(descriptor_1, descriptor_1);
        cv::threshold(descriptor_1, descriptor_1, 0.2, 0.2, cv::THRESH_TRUNC);
        cv::normalize(descriptor_1, descriptor_1);
    }

    // Process keypoints2
    descriptor2 = cv::Mat::zeros(keypoints2.size(), descriptorSize, CV_32F);
    for (size_t i = 0; i < keypoints2.size(); ++i)
    {
        cv::Mat descriptor_2 = descriptor2.row(i);
        const cv::KeyPoint& kp = keypoints2[i];

        for (int x = -8; x <= 7; ++x)
        {
            for (int y = -8; y <= 7; ++y)
            {
                float theta = std::atan2(kp.pt.y -  y, kp.pt.x - x) * (180.0 / CV_PI);
                float phi = 0.0; // Since there is no z component

                int thetaBin = static_cast<int>(theta / (360.0 / numThetaBins));
                int phiBin = static_cast<int>((phi + 90.0) / (180.0 / numPhiBins));
                int gradientBin = static_cast<int>(M / (M_max / numGradientBins));

                int index = thetaBin * numPhiBins * numGradientBins + phiBin * numGradientBins + gradientBin;
                descriptor_2.at<float>(0, index) += M;
            }
        }

        cv::normalize(descriptor_2, descriptor_2);
        cv::threshold(descriptor_2, descriptor_2, 0.2, 0.2, cv::THRESH_TRUNC);
        cv::normalize(descriptor_2, descriptor_2);
    }
}


void HyperFunctions::DimensionalityReduction()
{
    // this is a precursor for feature extraction
    // reduces the dimensionality of the data to a single greyscale image

    if (dimensionality_reduction == 0)
    {
        // cout<<"dimensionality reduction not needed"<<endl;
    }
    else if (dimensionality_reduction == 1)
    {
        gaSpace(true);
        feature_img1 = ga_img;
        gaSpace(false);
        feature_img2 = ga_img;
    }
    else if (dimensionality_reduction == 2)
    {
        PCA_img(true);
        feature_img1 = pca_img;
        PCA_img(false);
        feature_img2 = pca_img;
    }
}

//Stitching two images together
void HyperFunctions::Stitching(){
    // feature images must be 8uc1 not 8uc3 ie greyscale and not color images
    
    bool use_homography=false;

    // filter matched points
    //calculation of max and min distances between keypoints
    int movementDirection = 0;
    double max_dist = 0; double min_dist = 100;
    for (const auto& m : matches)
    {
        double dist = m.distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    for (const auto& m : matches)
    {
        if (m.distance <= 1.5 * min_dist)
        {
            good_point1.push_back(keypoints1.at(m.queryIdx).pt);
            good_point2.push_back(keypoints2.at(m.trainIdx).pt);
        }
    }

    if (use_homography)
    {
        // below is not correct, still a work in progress
        Mat h = findHomography( good_point1, good_point2, RANSAC );
        // Use homography to warp image
        Mat img1Warped;
        warpPerspective(feature_img1, img1Warped, h, feature_img2.size());
        Mat result;
        feature_img2.copyTo(result);
        img1Warped.copyTo(result, feature_img2);
        stitch_img=result;
    }
    else
    {
       
        
        cv::Rect croppImg1(0, 0, feature_img1.cols, feature_img1.rows);
        cv::Rect croppImg2(0, 0, feature_img2.cols, feature_img2.rows); 


        // movementDirection tells us are both the images aligned or not if not adjust the images accordingly.
        int imgWidth = feature_img1.cols;
        for (int i = 0; i < good_point1.size(); ++i)
        {
            if (good_point1[i].x < imgWidth)
            {
                croppImg1.width = good_point1.at(i).x;
                croppImg2.x = good_point2[i].x;
                croppImg2.width = feature_img2.cols - croppImg2.x;
                movementDirection = good_point1[i].y - good_point2[i].y;
                imgWidth = good_point1[i].x;
            }
        }
        Mat image1 = feature_img1(croppImg1);
        Mat image2 = feature_img2(croppImg2);

       
        int maxHeight = image1.rows > image2.rows ? image1.rows : image2.rows;
        int maxWidth = image1.cols + image2.cols;
        stitch_img=cv::Mat::zeros(cv::Size(maxWidth, maxHeight + abs(movementDirection)), CV_8UC1);
        if (movementDirection > 0)
        {
            cv::Mat half1(stitch_img, cv::Rect(0, 0, image1.cols, image1.rows));
            image1.copyTo(half1);
            cv::Mat half2(stitch_img, cv::Rect(image1.cols, abs(movementDirection),image2.cols, image2.rows));
            image2.copyTo(half2);
        }
        else
        {
            cv::Mat half1(stitch_img, cv::Rect(0, abs(movementDirection), image1.cols, image1.rows));
            image1.copyTo(half1);
            cv::Mat half2(stitch_img, cv::Rect(image1.cols,0 ,image2.cols, image2.rows));
            image2.copyTo(half2);
        }
        
    
    }

    Mat disp_stitch;
    cv::resize(stitch_img,disp_stitch,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR);
    imshow("Stitched Image", disp_stitch );
}
// Detects, describes, and matches keypoints between 2 feature images
void HyperFunctions::FeatureExtraction()
{
    // feature_detector=0; 0 is sift, 1 is surf, 2 is orb, 3 is fast
    // feature_descriptor=0; 0 is sift, 1 is surf, 2 is orb
    // feature_matcher=0; 0 is flann, 1 is bf
    // cout<<feature_detector<<" "<<feature_descriptor<<" "<<feature_matcher<<endl;

    if (feature_detector == 0 && feature_descriptor == 2)
    {
        cout << "invalid detector/descriptor combination" << endl;
        return;
    }

    // if (feature_detector < 0 || feature_detector > 4 || feature_descriptor < 0 || feature_descriptor > 3 || feature_matcher < 0 || feature_matcher > 1)
    // {
    //     cout << "invalid feature combination" << endl;
    // }

    int minHessian = 400;
    Ptr<SURF> detector_SURF = SURF::create(minHessian);
    cv::Ptr<SIFT> detector_SIFT = SIFT::create();
    Ptr<FastFeatureDetector> detector_FAST = FastFeatureDetector::create();
    Ptr<ORB> detector_ORB = ORB::create();
    Ptr<DescriptorMatcher> matcher;
    Mat descriptors1, descriptors2;

    // perform dimensionality reduction on the data to reduce hyperspectral image to a single layer
    // dimensionality_reduction=0; this is the variable that needs to be set
    // if set to 0, nothing is done, 1 is ga space, 2 is pca
    DimensionalityReduction();

    // feature_detector=0; 0 is sift, 1 is surf, 2 is orb, 3 is fast, 4 is SS-SIFT, 5 is custom
    if (feature_detector == 0)
    {
        detector_SIFT->detect(feature_img1, keypoints1);
        detector_SIFT->detect(feature_img2, keypoints2);
    }
    else if (feature_detector == 1)
    {
        detector_SURF->detect(feature_img1, keypoints1);
        detector_SURF->detect(feature_img2, keypoints2);
    }
    else if (feature_detector == 2)
    {
        detector_ORB->detect(feature_img1, keypoints1);
        detector_ORB->detect(feature_img2, keypoints2);
    }
    else if (feature_detector == 3)
    {
        detector_FAST->detect(feature_img1, keypoints1);
        detector_FAST->detect(feature_img2, keypoints2);
    }
    else if (feature_detector == 4)
    { // SS-SIFT feature detector
        SSDetector(feature_img1, keypoints1);
        SSDetector(feature_img2, keypoints2);

        // for (const auto& kp : keypoints1)
        // {
        //      std::cout << "x_1: " << kp.pt.x << ", y_1: " << kp.pt.y << std::endl;
        // }
        // for (const auto& kp : keypoints2)
        // {
        //      std::cout << "x_2: " << kp.pt.x << ", y_2: " << kp.pt.y << std::endl;
        // }
    }
    else if (feature_detector == 5)
    {
        // custom feature detector
        int spacing = 100;
        CreateCustomFeatureDetector(spacing, keypoints1, feature_img1); // input is the spacing between keypoints
        CreateCustomFeatureDetector(spacing, keypoints2, feature_img2);
    }
    // feature_descriptor=0; 0 is sift, 1 is surf, 2 is orb, 3 is SS-SIFT
    if (feature_descriptor == 0)
    {
        detector_SIFT->compute(feature_img1, keypoints1, descriptors1);
        detector_SIFT->compute(feature_img2, keypoints2, descriptors2);
    }
    else if (feature_descriptor == 1)
    {
        detector_SURF->compute(feature_img1, keypoints1, descriptors1);
        detector_SURF->compute(feature_img2, keypoints2, descriptors2);
    }
    else if (feature_descriptor == 2)
    {
        detector_ORB->compute(feature_img1, keypoints1, descriptors1);
        detector_ORB->compute(feature_img2, keypoints2, descriptors2);
    }
    else if (feature_descriptor == 3)
    {
        // SS-sift descriptor
        SSDescriptors(keypoints1, keypoints2, descriptors1, descriptors2, 1.0);
        // visualizeDescriptors(descriptors1);
        // visualizeDescriptors(descriptors2);
    }
    else if (feature_descriptor == 4)
    {
        // example descriptor 
        // HyperFunctions::computeCustomDescriptor ( const cv::Mat& feature_img, std::vector<cv::KeyPoint> & keypoints,cv::Mat& descriptors)
        computeCustomDescriptor ( feature_img1,  keypoints1, descriptors1);
        computeCustomDescriptor ( feature_img2,  keypoints2, descriptors2);
        
    }

    // feature_matcher=0; 0 is flann, 1 is bf
    if (feature_matcher == 0)
    {
        if (feature_descriptor == 2) // binary descriptor
        {
            matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
            matcher->match(descriptors1, descriptors2, matches);
        }
        else
        {
            matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
            matcher->match(descriptors1, descriptors2, matches);
        }
    }
    else if (feature_matcher == 1)
    {
        if (feature_descriptor == 2) // binary descriptor
        {
            matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
            matcher->match(descriptors1, descriptors2, matches);
        }
        else
        {
            matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
            matcher->match(descriptors1, descriptors2, matches);
        }
    }


    // filter matches with desired approach
    filter_matches(matches);

    
    Mat temp_img;
    drawMatches(feature_img1, keypoints1, feature_img2, keypoints2, matches, temp_img);

    cv::resize(temp_img, temp_img, Size(WINDOW_WIDTH * 2, WINDOW_HEIGHT), INTER_LINEAR);

    feature_img_combined = temp_img;
    imshow("Feature Images ", feature_img_combined);
}

void HyperFunctions::filter_matches(vector<DMatch> &matches)
{
    if (filter == 1)
    {
        // vector<Dmatch> good_matches;
        for (size_t i = 0; i < matches.size(); i++)
        {
            if (matches.at(i).distance > .75)
            {
                matches.erase(matches.begin() + i);
                i--;
            }
        }
    }
}

void HyperFunctions::computeCustomDescriptor ( const cv::Mat& feature_img, std::vector<cv::KeyPoint> & keypoints,cv::Mat& descriptors)
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



// Finds the transformation matrix between two images
void HyperFunctions::FeatureTransformation()
{
    // camera intrinsic parameters
    // double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

    // recovering the pose and the essential matrix
    Mat E, R, t, mask;
    vector<Point2f> points1;
    vector<Point2f> points2;
    for (int i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // uses ransac to filter out outliers
    E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
    // E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    //  recoverPose(E, points2, points1, R, t, focal, pp, mask);

    int inlier_num = 0;
    for (int i = 0; i < mask.rows; i++)
    {
        int temp_val2 = mask.at<uchar>(i);
        if (temp_val2 == 1)
        {
            inlier_num += 1;
        }
    }

    // cout<<" Essential Matrix"<<endl;
    // cout<<E<<endl;

    // cout << "Fundamental Matrix" << endl;
    // cout << R << endl
    //      << t << endl;

    // cout<<"inliers: " <<inlier_num<< " num of matches: "<<mask.rows<<endl;
    // cout<<" accuracy of feature matching: "<< (double)inlier_num/(double)(mask.rows)<<endl;
}

// To display classified image
void HyperFunctions::DispClassifiedImage()
{

    Mat temp_img;
    cv::resize(classified_img, temp_img, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    imshow("Classified Image", temp_img);
}

// To display false image (RGB layers are set by the user from the hyperspectral image)
void HyperFunctions::DispFalseImage()
{
    Mat temp_img;
    cv::resize(false_img, temp_img, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    imshow("False Image", temp_img);
}

// To display spectral similarity image
void HyperFunctions::DispSpecSim()
{
    Mat temp_img;
    cv::resize(spec_simil_img, temp_img, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    imshow("Spectral Similarity Image", temp_img);
}

// To display edge detection image (edges are from the classified image)
void HyperFunctions::DispEdgeImage()
{
    Mat temp_img;
    cv::resize(edge_image, temp_img, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    cv::imshow("Edge Detection Image", temp_img);
}

// Displays contour image (based on the classified image)
void HyperFunctions::DispContours()
{
    Mat temp_img;
    cv::resize(contour_img, temp_img, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    cv::imshow("Contour Image", temp_img);
}

// Displays the differences between the classified image and the contour image
void HyperFunctions::DispDifference()
{
    Mat temp_img;
    cv::resize(difference_img, temp_img, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    cv::imshow("Difference Image", temp_img);
}

// displays the tiled image (each image layer of the hyperspectral image is a tile)
void HyperFunctions::DispTiled()
{
    Mat temp_img;
    cv::resize(tiled_img, temp_img, Size(WINDOW_WIDTH, WINDOW_HEIGHT), INTER_LINEAR);
    cv::imshow("Tiled Image", temp_img);
}

// generates the false image by setting the RGB layers to what the user defines
void HyperFunctions::GenerateFalseImg()
{

    vector<Mat> channels(3);
    channels[0] = mlt1[false_img_b]; // b
    channels[1] = mlt1[false_img_g]; // g
    channels[2] = mlt1[false_img_r]; // r
    merge(channels, false_img);      // create new single channel image
}

//---------------------------------------------------------
// Name: DifferenceOfImages
// Description: Primarily for semantic interface tool to see how different parameters affect results.
// Outputs a binary image with black and white pixels.
// Black pixels represents no change between the filtered/approximated image and white pixels denotes a change.
//---------------------------------------------------------
void HyperFunctions::DifferenceOfImages()
{

    DetectContours();
    // create a copy of the incoming image in terms of size (length and width) and initialize as an all black image
    Mat output_image(classified_img.rows, classified_img.cols, CV_8UC1, cv::Scalar(0));
    // using 8 bit image so white pixel has a value of 255

    Vec3b temp_val, compare_val; // rgb value of image at a pixel

    for (int i = 0; i < classified_img.rows; i++)
    {
        for (int j = 0; j < classified_img.cols; j++)
        {
            if (classified_img.at<Vec3b>(i, j) != contour_img.at<Vec3b>(i, j))
            {
                output_image.at<uchar>(i, j) = 255;
            }
        }
    }

    difference_img = output_image;
}

// creates a binary image that sets boundary pixels as white and non-boundary pixels as black
// input is a classified image
// the output of this is used to find the contours in the image
// this is multi-threaded for speed requirements
void EdgeDetection_Child(int id, int i, Mat *output_image, Mat *classified_img2)
{

    bool edge = false;
    Vec3b temp_val, compare_val; // rgb value of image at a pixel
    Mat classified_img = *classified_img2;
    for (int j = 0; j < classified_img.cols; j++)
    {
        edge = false;

        if (i == 0 || j == 0 || i == classified_img.rows - 1 || j == classified_img.cols - 1)
        {
            // set boundaries of image to edge
            edge = true;
        }
        else
        {
            temp_val = classified_img.at<Vec3b>(i, j); // in form (y,x)

            // go through image pixel by pixel  and look at surrounding 8 pixels, if there is a difference in color between center and other pixels, then it is an edge

            for (int a = -1; a < 2; a++)
            {
                for (int b = -1; b < 2; b++)
                {
                    compare_val = classified_img.at<Vec3b>(i + a, j + b);
                    if (compare_val != temp_val)
                    {
                        edge = true;
                    }
                }
            }
        }

        if (edge)
        {
            // set edge pixel to white
            output_image->at<uchar>(i, j) = 255;
        }
    }
}

void HyperFunctions::EdgeDetection()
{
    // create a copy of the incoming image in terms of size (length and width) and initialize as an all black image
    Mat output_image(classified_img.rows, classified_img.cols, CV_8UC1, cv::Scalar(0));
    // using 8 bit image so white pixel has a value of 255

    ctpl::thread_pool p(num_threads);
    for (int i = 0; i < classified_img.rows; i++)
    {
        p.push(EdgeDetection_Child, i, &output_image, &classified_img);
    }

    edge_image = output_image;
}

// threaded function for DetectContours
void Classification_Child(int id, int i, Mat *classified_img, Mat *edge_image, vector<vector<Point>> *contours_approx, vector<Vec4i> *hierarchy, vector<Vec3b> *contour_class)
{
    Mat b_hist, g_hist, r_hist;
    int histSize = 256;
    float range[] = {0, 256}; // the upper boundary is exclusive
    const float *histRange[] = {range};
    bool uniform = true, accumulate = false;
    vector<Mat> bgr_planes;
    split(*classified_img, bgr_planes);
    Vec3b color_temp;

    Mat drawing2 = Mat::zeros(edge_image->size(), CV_8UC1);
    Scalar color = Scalar(255);
    drawContours(drawing2, *contours_approx, i, color, FILLED, 8, *hierarchy, 0, Point());
    calcHist(&bgr_planes[0], 1, 0, drawing2, b_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, drawing2, g_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, drawing2, r_hist, 1, &histSize, histRange, uniform, accumulate);
    int max_r = 0, max_b = 0, max_g = 0;
    int max_r_loc = 0, max_b_loc = 0, max_g_loc = 0;

    for (int j = 0; j < 256; j++)
    {
        if (r_hist.at<float>(j) > max_r)
        {
            max_r = r_hist.at<float>(j);
            max_r_loc = j;
            color_temp[1] = max_r_loc;
        }

        if (g_hist.at<float>(j) > max_g)
        {
            max_g = g_hist.at<float>(j);
            max_g_loc = j;
            color_temp[0] = max_g_loc;
        }

        if (b_hist.at<float>(j) > max_b)
        {
            max_b = b_hist.at<float>(j);
            max_b_loc = j;
            color_temp[2] = max_b_loc;
        }
    }
    (*contour_class)[i] = color_temp;
}

// Description: to identify and extract the boundaries (or contours) of specific objects
// in an image to make out the shapes of objects.
void HyperFunctions::DetectContours()
{

    // cout<<"min area "<<min_area<<" coeff poly "<<polygon_approx_coeff<<endl;
    if (edge_image.empty())
    {
        EdgeDetection();
    }

    read_spectral_json(spectral_database);

    contours_approx.clear();
    vector<Vec4i> hierarchy;
    double img_area_meters, img_area_pixels, contour_temp_area;
    img_area_pixels = edge_image.rows * edge_image.cols;
    img_area_meters = pow(double(2) * avgDist * tan(fov * 3.14159 / double(180) / (double)2), 2);

    findContours(edge_image, contours_approx, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    for (int i = 0; i < contours_approx.size(); i++)
    {
        contour_temp_area = img_area_meters * contourArea(contours_approx[i]) / img_area_pixels;
        if (contour_temp_area < min_area)
        {
            contours_approx[i].clear();
            contours_approx[i].push_back(Point(0, 0));
        }
    }

    Mat drawing = Mat::zeros(edge_image.size(), CV_8UC3);
    vector<Vec3b> contour_class(contours_approx.size() + 1);
    ctpl::thread_pool p(num_threads);
    for (int i = 0; i < contours_approx.size(); i++)
    {
        if (contours_approx[i].size() > 2)
        {
            p.push(Classification_Child, i, &classified_img, &edge_image, &contours_approx, &hierarchy, &contour_class);
        }
    }

    // wait until threadpool is finished here
    while (p.n_idle() < num_threads)
    {
        // cout<<" running threads "<< p.size()  <<" idle threads "<<  p.n_idle()  <<endl;
        // do nothing
    }

    // int count =0;
    for (int i = 0; i < contours_approx.size(); i++)
    {
        if (contours_approx[i].size() > 2)
        {

            Vec3b color = contour_class[i];

            string classification = "unknown";
            for (int j = 0; j < color_combos.size(); j++)
            {
                if (color == color_combos[j])
                {
                    classification = class_list[j];
                }
            }
            // cout<<i<<" here "<<color<<endl;
            if (contours_approx[i][0] == Point(0, 0) && contours_approx[i][1] == Point(0, edge_image.rows - 1) && contours_approx[i][2] == Point(edge_image.cols - 1, edge_image.rows - 1) && contours_approx[i][3] == Point(edge_image.cols - 1, 0))
            {
                // writeJSON(event, contours_approx, i, "ballpark", count);
                // count++;
                Scalar temp_col = Scalar(color[2], color[0], color[1]);
                drawContours(drawing, contours_approx, i, temp_col, FILLED, 8, hierarchy, 0, Point());
            }
            else
            {
                // double epsilon = polygon_approx_coeff/1000 * arcLength(contours_approx[i], true);
                //  opencv method of approximating polygons
                // approxPolyDP(contours_approx[i],contour_approx_new[i],epsilon,true);
                //  thick edge approximation algorithm
                thickEdgeContourApproximation(i);
                if (contour_class[hierarchy[i][3]] != contour_class[i])
                {
                    // writeJSON(event, contours_approx, i, classification,count);
                    // count++;
                    Scalar temp_col = Scalar(color[2], color[0], color[1]);
                    drawContours(drawing, contours_approx, i, temp_col, FILLED, 8, hierarchy, 0, Point());
                }
            }
        }
    }

    // uncomment to write contours to json file
    // writeJSON_full(contours_approx, contour_class, hierarchy);
    contour_img = drawing;

} // end function

// Creates tile image or default/base image
// assumes 164 layers in hyperspectral image
void HyperFunctions::TileImage()
{
    Mat empty_img = mlt1[0] * 0;
    int num_chan = mlt1.size();
    int num_tile_rows = ceil(sqrt(num_chan));
    int cur_lay = 0;
    vector<Mat> matArrayRows;
    Mat matArray[num_chan];

    for (int i = 0; i < num_tile_rows; i++)
    {
        for (int j = 0; j < num_tile_rows; j++)
        {
            if (cur_lay < num_chan)
            {
                matArray[j] = mlt1[cur_lay];
            }
            else
            {
                matArray[j] = empty_img;
            }
            cur_lay++;
        }

        Mat temp_row;
        hconcat(matArray, num_tile_rows, temp_row);
        matArrayRows.push_back(temp_row);
    }
    for (int i = 0; i < num_tile_rows; i++)
    {
        matArray[i] = matArrayRows[i];
    }

    Mat temp_tile;
    vconcat(matArray, num_tile_rows, temp_tile);

    tiled_img = temp_tile;

    /*Mat empty_img= mlt1[0]*0;
    Mat h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13, base_image;

    // 13 x 13 tile image singe 164 bands
    Mat matArray1[]={mlt1[0],mlt1[1],mlt1[2],mlt1[3],mlt1[4],mlt1[5],mlt1[6],mlt1[7],mlt1[8],mlt1[9],mlt1[10],mlt1[11],mlt1[12]};
    Mat matArray2[]={mlt1[13],mlt1[14],mlt1[15],mlt1[16],mlt1[17],mlt1[18],mlt1[19],mlt1[20],mlt1[21],mlt1[22],mlt1[23],mlt1[24],mlt1[25]};
    Mat matArray3[]={mlt1[26],mlt1[27],mlt1[28],mlt1[29],mlt1[30],mlt1[31],mlt1[32],mlt1[33],mlt1[34],mlt1[35],mlt1[36],mlt1[37],mlt1[38]};
    Mat matArray4[]={mlt1[39],mlt1[40],mlt1[41],mlt1[42],mlt1[43],mlt1[44],mlt1[45],mlt1[46],mlt1[47],mlt1[48],mlt1[49],mlt1[50],mlt1[51]};
    Mat matArray5[]={mlt1[52],mlt1[53],mlt1[54],mlt1[55],mlt1[56],mlt1[57],mlt1[58],mlt1[59],mlt1[60],mlt1[61],mlt1[62],mlt1[63],mlt1[64]};
    Mat matArray6[]={mlt1[65],mlt1[66],mlt1[67],mlt1[68],mlt1[69],mlt1[70],mlt1[71],mlt1[72],mlt1[73],mlt1[74],mlt1[75],mlt1[76],mlt1[77]};
    Mat matArray7[]={mlt1[78],mlt1[79],mlt1[80],mlt1[81],mlt1[82],mlt1[83],mlt1[84],mlt1[85],mlt1[86],mlt1[87],mlt1[88],mlt1[89],mlt1[90]};
    Mat matArray8[]={mlt1[91],mlt1[92],mlt1[93],mlt1[94],mlt1[95],mlt1[96],mlt1[97],mlt1[98],mlt1[99],mlt1[100],mlt1[101],mlt1[102],mlt1[103]};
    Mat matArray9[]={mlt1[104],mlt1[105],mlt1[106],mlt1[107],mlt1[108],mlt1[109],mlt1[110],mlt1[111],mlt1[112],mlt1[113],mlt1[114],mlt1[115],mlt1[116]};
    Mat matArray10[]={mlt1[117],mlt1[118],mlt1[119],mlt1[120],mlt1[121],mlt1[122],mlt1[123],mlt1[124],mlt1[125],mlt1[126],mlt1[127],mlt1[128],mlt1[129]};
    Mat matArray11[]={mlt1[130],mlt1[131],mlt1[132],mlt1[133],mlt1[134],mlt1[135],mlt1[136],mlt1[137],mlt1[138],mlt1[139],mlt1[140],mlt1[141],mlt1[142]};
    Mat matArray12[]={mlt1[143],mlt1[144],mlt1[145],mlt1[146],mlt1[147],mlt1[148],mlt1[149],mlt1[150],mlt1[151],mlt1[152],mlt1[153],mlt1[154],mlt1[155]};
    Mat matArray13[]={mlt1[156],mlt1[157],mlt1[158],mlt1[159],mlt1[160],mlt1[161],mlt1[162],mlt1[163],empty_img,empty_img,empty_img,empty_img,empty_img};

    // concatenates the rows of images
    hconcat(matArray1,13,h1);
    hconcat(matArray2,13,h2);
    hconcat(matArray3,13,h3);
    hconcat(matArray4,13,h4);
    hconcat(matArray5,13,h5);
    hconcat(matArray6,13,h6);
    hconcat(matArray7,13,h7);
    hconcat(matArray8,13,h8);
    hconcat(matArray9,13,h9);
    hconcat(matArray10,13,h10);
    hconcat(matArray11,13,h11);
    hconcat(matArray12,13,h12);
    hconcat(matArray13,13,h13);
    Mat matArray14[]={h1,h2,h3,h4,h5 ,h6,h7,h8,h9,h10,h11,h12,h13 };
    vconcat(matArray14,13,base_image);

    tiled_img=base_image;*/
}

//---------------------------------------------------------
// Name: read_spectral_json
// Description: reads json file containing spectral information and RGB color values
// for creating a classified image. used in Image Tool.
//---------------------------------------------------------
void HyperFunctions::read_spectral_json(string file_name)
{

    // read spectral database and return classes and rgb values

    Vec3b color;

    vector<string> class_list2;
    color_combos.clear();

    ifstream ifs(file_name);
    Json::Reader reader;
    Json::Value completeJsonData;
    reader.parse(ifs, completeJsonData);
    // cout<< "Complete JSON data: "<<endl<< completeJsonData<<endl;

    // load rgb values and classes

    for (auto const &id3 : completeJsonData["Color_Information"].getMemberNames())
    {
        color[2] = completeJsonData["Color_Information"][id3]["red_value"].asInt();
        color[0] = completeJsonData["Color_Information"][id3]["blue_value"].asInt();
        color[1] = completeJsonData["Color_Information"][id3]["green_value"].asInt();

        // cout<<id3<<color<<endl;
        color_combos.push_back(color);
        class_list2.push_back(id3);
    }

    class_list = class_list2;
}

//---------------------------------------------------------
// Name: writeJSON
// Description: holds information about the extracted contours for navigation
//---------------------------------------------------------
void HyperFunctions::writeJSON(Json::Value &event, vector<vector<Point>> &contours, int idx, string classification, int count)
{

    Json::Value vec(Json::arrayValue);
    for (int i = 0; i < contours[idx].size(); i++)
    {
        Json::Value arr(Json::arrayValue);
        // cout << (contours[idx][i].x) << endl;
        // cout << (contours[idx][i].y) << endl;
        arr.append(contours[idx][i].x);
        arr.append(contours[idx][i].y);
        vec.append(arr);
    }

    // change below to the correct classification
    string Name;
    if (classification == "ballpark")
        Name = "Ballpark";
    else
        Name = classification + to_string(count);
    event["features"][count]["type"] = "Feature";
    event["features"][count]["properties"]["Name"] = Name;
    event["features"][count]["properties"]["sensor_visibility_above"] = "yes";
    event["features"][count]["properties"]["sensor_visibility_side"] = "yes";
    event["features"][count]["properties"]["traversability_av"] = "100";
    event["features"][count]["properties"]["traversability_gv"] = "100";
    event["features"][count]["properties"]["traversability_ped"] = "100";
    event["features"][count]["geometry"]["type"] = "LineString";
    event["features"][count]["geometry"]["coordinates"] = vec;
}

//---------------------------------------------------------
// Name: writeJSON
// Description: holds information about the extracted contours for navigation
//---------------------------------------------------------
void HyperFunctions::writeJSON_full(vector<vector<Point>> contours, vector<Vec3b> contour_class, vector<Vec4i> hierarchy)
{

    std::ofstream file_id;
    file_id.open(output_polygons);
    Json::Value event;
    // initialise JSON file
    event["type"] = "FeatureCollection";
    event["generator"] = "Img Segmentation";
    string Name;

    int count = 0;
    int idx = 0;
    string classification = class_list[idx];

    for (int idx = 0; idx < contours.size(); idx++)
    {
        bool write_to_file = false;
        if (contours[idx].size() > 2 && contour_class[hierarchy[idx][3]] != contour_class[idx] && idx > 0)
        {
            Vec3b color = contour_class[idx];
            classification = "unknown";
            for (int j = 0; j < color_combos.size(); j++)
            {
                if (color == color_combos[j])
                {
                    classification = class_list[j];
                }
            }

            write_to_file = true;
        }
        // else if (contours[idx][0]==Point(0,0) && contours[idx][1]==Point(0,edge_image.rows-1)  && contours[idx][2]==Point(edge_image.cols-1,edge_image.rows-1)  && contours[idx][3]==Point(edge_image.cols-1,0))
        else if (idx == 0)
        {
            Name = "Ballpark";
            write_to_file = true;
        }

        if (write_to_file)
        {

            Json::Value vec(Json::arrayValue);
            for (int i = 0; i < contours[idx].size(); i++)
            {
                Json::Value arr(Json::arrayValue);
                arr.append(contours[idx][i].x);
                arr.append(contours[idx][i].y);
                vec.append(arr);
            }

            if (idx > 0)
            {
                Name = classification + to_string(count);
            }
            // cout<<Name<<" "<<idx<<endl;
            event["features"][count]["type"] = "Feature";
            event["features"][count]["properties"]["Name"] = Name;
            event["features"][count]["properties"]["sensor_visibility_above"] = "yes";
            event["features"][count]["properties"]["sensor_visibility_side"] = "yes";
            event["features"][count]["properties"]["traversability_av"] = "100";
            event["features"][count]["properties"]["traversability_gv"] = "100";
            event["features"][count]["properties"]["traversability_ped"] = "100";
            event["features"][count]["geometry"]["type"] = "LineString";
            event["features"][count]["geometry"]["coordinates"] = vec;
            count++;
        }
    }

    Json::StyledWriter styledWriter;
    file_id << styledWriter.write(event);
    file_id.close();
}

//---------------------------------------------------------
// Name: read_img_json
// Description: obtains info about the camera/image to help convert pixel coordinates to GPS coordinates.
//---------------------------------------------------------
void HyperFunctions::read_img_json(string file_name)
{

    ifstream ifs(file_name);
    Json::Reader reader;
    Json::Value completeJsonData;
    reader.parse(ifs, completeJsonData);

    fov = completeJsonData["FOV"].asDouble();
    avgDist = completeJsonData["AvgDistanceMeters"].asDouble();
    gps1 = completeJsonData["GPS1"].asDouble();
    gps2 = completeJsonData["GPS2"].asDouble();
}

// saves spectral and color information to json file of spectral curves
// assumes a ultris x20 hyperspectral image
// Accesses camera information (camera_database) and modifies spectral database
void HyperFunctions::save_ref_spec_json(string item_name)
{
    int img_hist[mlt1.size()];
    for (int i = 0; i < mlt1.size(); i++)
    {
        img_hist[i] = mlt1[i].at<uchar>(cur_loc);
    }
    string user_input = item_name;

    // modify spectral database
    ifstream ifs2(spectral_database);
    Json::Reader reader2;
    Json::Value completeJsonData2;
    reader2.parse(ifs2, completeJsonData2);

    std::ofstream file_id;
    file_id.open(spectral_database);
    Json::Value value_obj;
    value_obj = completeJsonData2;
    // save histogram to json file

    for (int i = 0; i < mlt1.size(); i++)
    {
        string zero_pad_result;

        if (i < 10)
        {
            zero_pad_result = "000" + to_string(i);
        }
        else if (i < 100)
        {
            zero_pad_result = "00" + to_string(i);
        }
        else if (i < 1000)
        {
            zero_pad_result = "0" + to_string(i);
        }
        else if (i < 10000)
        {
            zero_pad_result = to_string(i);
        }
        else
        {
            cout << " error: out of limit for spectral wavelength" << endl;
            return;
        }

        value_obj["Spectral_Information"][user_input][zero_pad_result] = img_hist[i];
    }

    // change to 163, 104, 64 layer value, order is bgr

    if (mlt1.size() == 164)
    {
        value_obj["Color_Information"][user_input]["red_value"] = img_hist[64];
        value_obj["Color_Information"][user_input]["blue_value"] = img_hist[104];
        value_obj["Color_Information"][user_input]["green_value"] = img_hist[163];
    }
    else
    {
        value_obj["Color_Information"][user_input]["red_value"] = img_hist[1 * mlt1.size() / 3];
        value_obj["Color_Information"][user_input]["blue_value"] = img_hist[2 * mlt1.size() / 3];
        value_obj["Color_Information"][user_input]["green_value"] = img_hist[3 * mlt1.size() / 3 - 1];
    }
    // write out to json file
    Json::StyledWriter styledWriter;
    file_id << styledWriter.write(value_obj);
    file_id.close();

    /*int img_hist[mlt1.size()-1];
    for (int i=0; i<=mlt1.size()-1;i++)
    {
        img_hist[i]=mlt1[i].at<uchar>(cur_loc);
    }

    string user_input=item_name;
    ifstream ifs(camera_database );
    Json::Reader reader;
    Json::Value completeJsonData;
    reader.parse(ifs,completeJsonData);
    int min_wave, spect_step,  max_wave, loop_it=0 ;
    max_wave = completeJsonData["Ultris_X20"]["Camera_Information"]["Max_Wavelength"].asInt();
    min_wave = completeJsonData["Ultris_X20"]["Camera_Information"]["Min_Wavelength"].asInt();
    spect_step = completeJsonData["Ultris_X20"]["Camera_Information"]["Spectral_Sampling"].asInt();

    // modify spectral database
    ifstream ifs2(spectral_database);
    Json::Reader reader2;
    Json::Value completeJsonData2;
    reader.parse(ifs2,completeJsonData2);

    std::ofstream file_id;
    file_id.open(spectral_database);
    Json::Value value_obj;
    value_obj = completeJsonData2;
    // save histogram to json file
    for (int i=min_wave; i<=(max_wave);i+=spect_step)
    {
    value_obj["Spectral_Information"][user_input][to_string(i)] = img_hist[loop_it];
    loop_it+=1;
    }

    // change to 163, 104, 64 layer value, order is bgr
    value_obj["Color_Information"][user_input]["red_value"]=img_hist[64];
    value_obj["Color_Information"][user_input]["blue_value"]=img_hist[104];
    value_obj["Color_Information"][user_input]["green_value"]=img_hist[163];
    // write out to json file
    Json::StyledWriter styledWriter;
    file_id << styledWriter.write(value_obj);
    file_id.close();
    */
}

// reads spectral and color information from items in json file
void HyperFunctions::read_ref_spec_json(string file_name)
{
    // read json file
    ifstream ifs2(file_name);
    Json::Reader reader2;
    Json::Value completeJsonData2;
    reader2.parse(ifs2, completeJsonData2);
    vector<string> class_list2;

    // initialize variables
    int layer_values;
    Vec3b color;
    if (reference_spectrums.size() > 0)
    {
        reference_spectrums.clear();
    }
    if (color_combos.size() > 0)
    {
        color_combos.clear();
    }

    // gets spectrum of items in spectral database
    for (auto const &id : completeJsonData2["Spectral_Information"].getMemberNames())
    {
        vector<int> tempValues1;
        for (auto const &id2 : completeJsonData2["Spectral_Information"][id].getMemberNames())
        {
            layer_values = completeJsonData2["Spectral_Information"][id][id2].asInt();
            tempValues1.push_back(layer_values);
        }
        reference_spectrums.push_back(tempValues1);
    }

    // gets colors of items in database
    for (auto const &id3 : completeJsonData2["Color_Information"].getMemberNames())
    {
        color[0] = completeJsonData2["Color_Information"][id3]["red_value"].asInt();
        color[1] = completeJsonData2["Color_Information"][id3]["blue_value"].asInt();
        color[2] = completeJsonData2["Color_Information"][id3]["green_value"].asInt();
        color_combos.push_back(color);
        class_list2.push_back(id3);
    }
    
    class_list = class_list2;
}

// creates a blank spectral database to fill in
void HyperFunctions::save_new_spec_database_json()
{
    std::ofstream file_id3;
    file_id3.open(spectral_database);
    Json::Value new_obj;
    Json::StyledWriter styledWriter2;
    new_obj["Spectral_Information"] = {};
    new_obj["Color_Information"] = {};
    file_id3 << styledWriter2.write(new_obj);
    file_id3.close();
}

//---------------------------------------------------------
// Name: SemanticSegmenter
// Description: Takes hyperspectral data and assigns each pixel a color
// based on which reference spectra it is most similar to.
//---------------------------------------------------------
void HyperFunctions::SemanticSegmenter()
{

    // classified_img

    vector<Mat> temp_results;

    for (int i = 0; i < reference_spectrums.size(); i++)
    {
        ref_spec_index = i;
        this->SpecSimilParent();
        temp_results.push_back(spec_simil_img);
    }
    Mat temp_class_img(mlt1[1].rows, mlt1[1].cols, CV_8UC3, Scalar(0, 0, 0));

    for (int k = 0; k < mlt1[1].cols; k++)
    {
        for (int j = 0; j < mlt1[1].rows; j++)
        {
            double low_val;
            for (int i = 0; i < temp_results.size(); i++)
            {
                if (i == 0)
                {
                    low_val = temp_results[i].at<uchar>(j, k);
                    if (low_val <= classification_threshold)
                    {
                        temp_class_img.at<Vec3b>(Point(k, j)) = color_combos[i];
                    }
                }
                else
                {
                    if (temp_results[i].at<uchar>(j, k) < low_val && temp_results[i].at<uchar>(j, k) <= classification_threshold)
                    {
                        low_val = temp_results[i].at<uchar>(j, k);
                        temp_class_img.at<Vec3b>(Point(k, j)) = color_combos[i];
                    }
                }
            }
        }
    }

    classified_img = temp_class_img;
}

void SpecSimilChild(int threadId, int algorithmId, int columnIndex, vector<Mat> *mlt, vector<int> *reference_spectrum_ptr, Mat *outputSimilarityImage, double *max_sim_val, double *min_sim_val, bool tune_spec_sim)
{

    vector<Mat> hyperspectralImage = *mlt; // dereferences
    vector<int> reference_spectrumAsInt = *reference_spectrum_ptr;
    vector<double> reference_spectrum(reference_spectrumAsInt.begin(), reference_spectrumAsInt.end());

    // Normalizes the reference vector if that is necessary for the comparison algorithm
    // Some algorithms re-normalize anyway which is a source of future optimizations (get rid of redundant code)
    // needs to be done in parent so not repeated
    // if (algorithmId == 4 || algorithmId == 6 || algorithmId == 7)
    // {
    //     double reference_spectrum_sum = 0;
    //     for (int i = 0; i < reference_spectrum.size(); i++)
    //     {
    //         reference_spectrum_sum += reference_spectrum[i];
    //     }
    //     for (int i = 0; i < reference_spectrum.size(); i++)
    //     {
    //         reference_spectrum[i] /= reference_spectrum_sum;
    //     }
    // }

    for (int rowIndex = 0; rowIndex < hyperspectralImage[1].rows; rowIndex++)
    {
        // Find the pixel spectrum
        vector<double> pixel_spectrum;
        // double pixel_spectrum_sum = 0;

        for (int layer = 0; layer < reference_spectrum.size(); layer++) // Assumes that pixel and reference spectra are the same size.
        {
            pixel_spectrum.push_back(hyperspectralImage[layer].at<uchar>(rowIndex, columnIndex));
            // pixel_spectrum_sum += hyperspectralImage[layer].at<uchar>(rowIndex, columnIndex);
        }

        // // Normalizes the pixel vector if that is necessary for the comparison algorithm
        // if (algorithmId == 4 || algorithmId == 6 || algorithmId == 7)
        // {
        //     for (int layer = 0; layer < reference_spectrum.size(); layer++)
        //     {
        //         pixel_spectrum[layer] /= pixel_spectrum_sum;
        //     }
        // }

        double similarityValue = 0;
        double scaling_factor; 
        switch(algorithmId) { //Manipulation of similarity values not complete yet...
        // most of these having scaling parameters that need to be tuned to the environment and desired level of spectral discrimination
            case 0:
                similarityValue = calculateSAM(reference_spectrum, pixel_spectrum) * 81.169; //81.169=255/3.141; 
                // 0-255 is range of values for 8 bit image for visualization, 0-pi is range of sam values
                //Below is equivalent using the calculateCOS function
                //similarityValue = acos(calculateCOS(reference_spectrum, pixel_spectrum)) / 3.141592 * 255;
                break;
            case 1:
                similarityValue = (1-calculateSCM(reference_spectrum, pixel_spectrum)) * 127.5 ;// 127.5=0.5 * 255;
                // range for scm is -1 to 1 (-1 is inverse similarity, 1 is similarity )
                break;
            case 2:
                scaling_factor = 100; // random scaling factor
                similarityValue = calculateSID(reference_spectrum, pixel_spectrum) * scaling_factor; // random scaling factor
                break;
            case 3:
                //similarityValue = calculateEUD(reference_spectrum, pixel_spectrum) / (reference_spectrum.size() + 255) * 255;
                scaling_factor = 0.10; // random scaling factor
                similarityValue = calculateEUD(reference_spectrum, pixel_spectrum) * scaling_factor;
                break;
            case 4:
                scaling_factor = 1; //.20;
                // this is not really chi square but leads to decent results 
                // it is similar to the canberra distance, but squares the numerator 
                similarityValue = calculateCsq(reference_spectrum, pixel_spectrum) * scaling_factor;
                break;
            case 5:
                //calculateCOS gives high values for things that are similar, so this flips that relationship
                scaling_factor = 255;
                similarityValue = (1-calculateCOS(reference_spectrum, pixel_spectrum)) * scaling_factor;
                break;
            case 6:
                //similarityValue = (calculateCB(reference_spectrum, pixel_spectrum) / (reference_spectrum.size() + 255)) * 255;
                scaling_factor = 1.0 /(reference_spectrum.size())  ;
                similarityValue = calculateCB(reference_spectrum, pixel_spectrum)   *  scaling_factor;
                break;
            case 7:
                scaling_factor = 127;
                similarityValue = calculateJM(reference_spectrum, pixel_spectrum) * scaling_factor;
                break;
            case 8: //Testing NS3
                scaling_factor = 800;
                similarityValue = scaling_factor * 
                    sqrt(
                            pow(sqrt(1/reference_spectrum.size()) * calculateEUD(reference_spectrum, pixel_spectrum), 2)
                            + pow(1-cos(calculateSAM(reference_spectrum, pixel_spectrum)), 2)
                        );
                break;
            case 9: //Testing JM-SAM
                scaling_factor = 255;
                similarityValue = scaling_factor * (calculateJM(reference_spectrum, pixel_spectrum) * tan(calculateSAM(reference_spectrum, pixel_spectrum)));
                break;
            case 10: //SCA
                scaling_factor =  81.169;
                similarityValue = scaling_factor  * acos((calculateSCM(reference_spectrum, pixel_spectrum)+1)/2);
                break;
            case 11: //SID-SAM
                // can have negative values, not sure of meaning
                scaling_factor = 255;
                similarityValue = scaling_factor * calculateSID(reference_spectrum, pixel_spectrum) * tan(calculateSAM(reference_spectrum, pixel_spectrum));
                break;
            case 12: //SID-SCA
                scaling_factor = 255;
                similarityValue = scaling_factor * calculateSID(reference_spectrum, pixel_spectrum) * tan( acos((calculateSCM(reference_spectrum, pixel_spectrum)+1)/2));
                break;
            case 13: //Hellinger Distance
                scaling_factor = 255;
                similarityValue = calculateHDist(reference_spectrum, pixel_spectrum) * scaling_factor;
                break;
            case 14: //Canberra distance
                scaling_factor = 0.8;
                similarityValue = scaling_factor * calculateCanb(reference_spectrum, pixel_spectrum);
                break;
        }


        if (tune_spec_sim)
        {
           if (similarityValue > *max_sim_val)
            {
                *max_sim_val = similarityValue;
            }
            else if (similarityValue < *min_sim_val)
            {
                *min_sim_val = similarityValue;
            }
        }

        // make sure in acceptable range
        if (similarityValue > 255)
        {
            *max_sim_val = similarityValue;
            similarityValue = 255;
        }
        else if (similarityValue < 0)
        {
            *min_sim_val = similarityValue;
            similarityValue = 0;
        }

        outputSimilarityImage->at<uchar>(rowIndex, columnIndex) = similarityValue; 
    }
}

//---------------------------------------------------------
// Name: SpecSimilParent
// Description: to determine the similarity between sets
// of data (spectral curves) within threadpool based on their spectral properties
//---------------------------------------------------------
void HyperFunctions::SpecSimilParent()
{
    Mat temp_img(mlt1[1].rows, mlt1[1].cols, CV_8UC1, Scalar(0));
    spec_simil_img = temp_img;

    ctpl::thread_pool p(num_threads);
    max_sim_val=0;
	min_sim_val=255;
    for (int k = 0; k < mlt1[1].cols; k += 1)
    {
        p.push(SpecSimilChild, spec_sim_alg, k, &mlt1, &reference_spectrums[ref_spec_index], &spec_simil_img, &max_sim_val, &min_sim_val, tune_spec_sim);
    }

    p.stop(true);

    // okay to have above 255 in order to improve spec sim contrast

    if (/*max_sim_val >255  &&*/ tune_spec_sim)
    {
        cout<<"alg: "<<spec_sim_alg<< " max sim val: "<<max_sim_val<<endl;
       
    }
    if ( /*min_sim_val <0 &&*/ tune_spec_sim)
    {
         cout<<"alg: "<<spec_sim_alg<<" min sim val: "<<min_sim_val<<endl;
    }

}

void HyperFunctions::thickEdgeContourApproximation(int idx)
{

    int sz = contours_approx[idx].size();
    int endPt = 2;
    int stPt = 1 - 1;
    int midPt = 1;
    vector<vector<int>> breakPt;

    while (stPt < sz && (stPt + endPt) < sz)
    {

        double xS = (double)contours_approx[idx][stPt].x;
        double yS = (double)contours_approx[idx][stPt].y;
        double xE = (double)contours_approx[idx][stPt + endPt].x;
        double yE = (double)contours_approx[idx][stPt + endPt].y;
        double xM = (double)contours_approx[idx][stPt + midPt].x;
        double yM = (double)contours_approx[idx][stPt + midPt].y;

        if (stPt == (sz - 1) || (stPt + endPt) == (sz - 1) || (stPt + midPt) == (sz - 1))
        {
            breakPt.push_back({stPt, 0});
            break;
        }

        double num = pow(((xM - xS) * (yE - yS) - (yM - yS) * (xE - xS)), 2.0);
        double den = sqrt(((xS - xE) * (xS - xE) + (yS - yE) * (yS - yE)));
        double dist = num / den;

        if (abs(dist) < polygon_approx_coeff)
        {
            endPt = endPt + 1;
            midPt = midPt + 1;
        }
        else
        {

            breakPt.push_back({stPt, stPt + endPt - 1});
            stPt = stPt + endPt - 1;
            endPt = 2;
            midPt = 1;
        }
    }
    vector<Point> contour_thickEdge;

    for (int i = 0; i < breakPt.size(); i++)
    {
        int idx1 = breakPt[i][0];
        int idx2 = breakPt[i][1];
        auto i1x = contours_approx[idx][idx1].x;
        auto i1y = contours_approx[idx][idx1].y;
        contour_thickEdge.push_back(Point(i1x, i1y));
    }
    int idx1 = breakPt[0][0];
    auto i1x = contours_approx[idx][idx1].x;
    auto i1y = contours_approx[idx][idx1].y;
    contour_thickEdge.push_back(Point(i1x, i1y));

    contours_approx[idx].clear();

    for (int i = 0; i < contour_thickEdge.size(); i++)
    {
        contours_approx[idx].push_back(Point(contour_thickEdge[i].x, contour_thickEdge[i].y));
    }

    int siz = contours_approx[idx].size();
}

// references  https://docs.opencv.org/3.4/d3/db0/samples_2cpp_2pca_8cpp-example.html
// https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html

static Mat formatImagesForPCA(const vector<Mat> &data)
{
    Mat dst(static_cast<int>(data.size()), data[0].rows * data[0].cols, CV_32F);
    for (unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1, 1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i, CV_32F);
    }
    return dst;
}

static Mat toGrayscale(InputArray _src)
{
    Mat src = _src.getMat();
    // only allow one channel
    if (src.channels() != 1)
    {
        CV_Error(Error::StsBadArg, "Only Matrices with one channel are supported");
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

void HyperFunctions::PCA_img(bool isImage1 = true)
{

    Mat data;
    vector<Mat> inputImage;
    if (isImage1)
    {
        data = formatImagesForPCA(mlt1);
        inputImage = mlt1;
    }
    else
    {
        data = formatImagesForPCA(mlt2);
        inputImage = mlt2;
    }
    int reduced_image_layers = 3;

    PCA pca(data, cv::Mat(), PCA::DATA_AS_ROW, reduced_image_layers);

    Mat principal_components = pca.eigenvectors;


    // below is for rgb image 
    vector<Mat> ReducedImage;
    for (int i = 0; i < reduced_image_layers; i++)
    {
        
        Mat point = pca.project(data.row(i));
        Mat reconstruction = pca.backProject(point);
       
        // cv::Size s = reconstruction.size();
        // int rows = s.height;
        // int cols = s.width;

        // std::cout << "Number of rows : " << rows << std::endl;
        // std::cout << "Number of cols : " << cols << std::endl;


        // Mat layer = principal_components.row(i)*data.t();
        // layer = pca.backProject(layer);
        Mat layer = reconstruction.reshape(inputImage[0].channels(), inputImage[0].rows);  // reshape from a row vector into image shape
        // Mat layer = principal_components * data.t();
        layer = toGrayscale(layer);
        ReducedImage.push_back(layer);
        // cout<<"layer size: "<<layer.size()<<endl;
        // cout<<"layer rows: "<<layer.rows<<endl;
        // cout<<"layer cols: "<<layer.cols<<endl;
        // cout<<"layer type: "<<layer.type()<<endl;
    }


    // imshow("PCA Results", ReducedImage);
    // imwritemulti(reduced_file_path,ReducedImage);

    vector<Mat> channels(3);
    Mat temp_false_img;
    channels[0] = ReducedImage[0]; // b
    // cout<<"0 size: "<<ReducedImage[0].size()<<endl;
    // cout<<"0 type: "<<ReducedImage[0].type()<<endl;
    // cout<<"0 channels: "<<ReducedImage[0].channels()<<endl;
    channels[1] = ReducedImage[1]; // g
    // cout<<"1 size: "<<ReducedImage[1].size()<<endl;
    // cout<<"1 type: "<<ReducedImage[1].type()<<endl;
    // cout<<"1 channels: "<<ReducedImage[1].channels()<<endl;
    channels[2] = ReducedImage[2]; // r
    // cout<<"2 size: "<<ReducedImage[2].size()<<endl;
    // cout<<"2 type: "<<ReducedImage[2].type()<<endl;
    // cout<<"2 channels: "<<ReducedImage[2].channels()<<endl;
    merge(channels, temp_false_img);      // create new single channel image
    // imshow("PCA Results RGB0", ReducedImage[0]);
    // imshow("PCA Results RGB1", ReducedImage[1]);
    // imshow("PCA Results RGB2", ReducedImage[2]);
    // cv::waitKey(0);
    pca_img = temp_false_img;
    // cout<<"pca_img size: "<<pca_img.size()<<endl;
    // cout<<"pca_img type: "<<pca_img.type()<<endl;
    // cout<<"pca_img channels: "<<pca_img.channels()<<endl;
    imwrite("../images/pca_result.png", pca_img);

    // // below is for grey scale image
    // // Demonstration of the effect of retainedVariance on the first image
    // Mat point = pca.project(data.row(0));                                                  // project into the eigenspace, thus the image becomes a "point"
    // Mat reconstruction = pca.backProject(point);                                           // re-create the image from the "point"
    // reconstruction = reconstruction.reshape(inputImage[0].channels(), inputImage[0].rows); // reshape from a row vector into image shape
    // reconstruction = toGrayscale(reconstruction);                                          // re-scale for displaying purposes
    // pca_img = reconstruction;
    // // not writing multiple layers yet because some versions of opencv do not have the function
    // // imwritemulti(reduced_file_path,reconstruction);
}
