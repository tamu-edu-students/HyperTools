#include <opencv2/opencv.hpp>
#include <gtk/gtk.h>
#include <iostream>
#include <cmath>
#include "../src/gtkfunctions.cpp"
#include "../src/hyperfunctions.cpp"


/*

1) Perform SIFT on the image

2) use spectral curve and look for peaks in areas where keypoints were located in step (1),
and that's where keypoints are likely located.



*/
// computing gradients for Descriptor theta phi
void computeGradient(const cv::Mat &src, cv::Mat &grad_x, cv::Mat grad_y, cv::Mat grad_z)
{
cv::Sobel(src, grad_x, CV_32F, 1, 0, 0, 3);
cv::Sobel(src, grad_y, CV_32F, 0, 1, 0, 3);
cv::Sobel(src, grad_z, CV_32F, 0, 0, 1, 3);

    // computing graidnet magnitude
    cv::Mat M;
    cv::magnitude(grad_x, grad_y, M);
    cv::magnitude(M, grad_z, M);
}
// Descriptor teta phi
void thetaPhi(const cv::Mat &grad_x, const cv::Mat &grad_y, const cv::Mat &grad_z)
{
    cv::Mat theta,phi;
    cv::phase(grad_x, grad_y, theta);                                     // Compute atan(grad_x/grad_y) and store it in theta
    cv::phase(grad_z, cv::Mat::zeros(grad_z.size(), grad_z.type()), phi); // Compute atan(grad_z/0) and store it in phi
    //return theta,phi; //Return theta & phi as descriptors
}

std::vector<cv::KeyPoint> performSift(const cv::Mat &hyperspectralCube, double sigma1, double sigma2, int octaveLevels, double k)
{
    const float M_max = 1.0; 
        cv::Mat hyperspectralCube; // we need a hyperspectralcube data

        std::vector<cv::KeyPoint> keypoints; // final keypoints vector

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
        

            std::vector<cv::KeyPoint> currentKeypoints;
            cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
            detector->detect(dog, currentKeypoints);
            // filtering the keypoints
            for (const cv::KeyPoint &keypoint : currentKeypoints)
            {
                if (keypoint.response > 0.75)
                {
                    keypoints.push_back(keypoint);
                }
            }
            // updating for the next octave
            previousScale = currentScale.clone();

            sigma1 *= k;
            sigma2 *= k;
        }
        return keypoints;
}


cv::Mat SsiftDescriptors(const std::vector<cv::KeyPoint> &keypoints, int numThetaBins, int numPhiBins, int numGradientBins, float M_max)
{
    const int numThetaBins = 8;
    const int numPhiBins = 4;
    const int numGradientBins = 8;
    const int descriptorSize = numThetaBins * numPhiBins * numGradientBins;
    float M;
    for (const cv::KeyPoint &keypoint : keypoints) // the descriptors
    {

        cv::Mat descriptor = cv::Mat::zeros(1, descriptorSize, CV_32F); // this is the descriptor for each iteration.

        for (int x = -8; x <= 7; ++x) // looping around the neighbors for each dimension
        {
            for (int y = -8; y <= 7; ++y)
            {
                for(int z =-4; z<=3;++z)
                {
                    float Gx, Gy, Gz; // gradients
                    float theta, phi; // theta and phi angle values

                    int thetaBin = static_cast<int>(theta / (360.0 / numThetaBins));
                    int phiBin = static_cast<int>((phi + 90.0) / (180.0 / numPhiBins));
                    int gradientBin = static_cast<int>(M / (M_max / numGradientBins));

                    
                    int index = thetaBin * numPhiBins * numGradientBins + phiBin * numGradientBins + gradientBin;
                    descriptor.at<float>(0, index) += M;
                }
            }
        }

       
        cv::normalize(descriptor, descriptor); // not sure why we need to do this.
        cv::threshold(descriptor, descriptor, 0.2, 0.2, cv::THRESH_TRUNC);
        cv::normalize(descriptor, descriptor);
       
       // storing the descriptors in a matrix
       for(const cv::KeyPoint &keypoint : keypoints)
       {
            cv::Mat descriptor = cv::Mat::zeros (1,descriptorSize, CV_32F);
            descriptor.push_back(descriptor);
       }

       
    }
    return descriptors;
}
