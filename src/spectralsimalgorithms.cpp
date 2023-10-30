#include <cmath>     // Include cmath for mathematical functions
#include <iostream>
#include <vector>    // Include vector for std::vector

using namespace std;

//Calculate the Spectral Angle Mapper algorithm
double calculateSAM(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum) {

    // Calculate the dot product of the two spectral vectors
    double dotProduct = 0.0;
    double magnitude1 = 0.0;
    double magnitude2 = 0.0;

    for (size_t i = 0; i < refSpectrum.size(); i++) {
        dotProduct += refSpectrum[i] * pixelSpectrum[i];
        magnitude1 += refSpectrum[i] * refSpectrum[i];
        magnitude2 += pixelSpectrum[i] * pixelSpectrum[i];
    }

    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);

    // Calculate the spectral angle
    double cosineSimilarity = dotProduct / (magnitude1 * magnitude2);
    double angle = std::acos(cosineSimilarity) / 3.141592;  // Angle in radians

    if (dotProduct<=0 || magnitude1<=0 || magnitude2<=0 ) //Investigate if we want this behavior
    {
        angle=1; // set to white due to an error
    }

    return angle;
}
//Calculate the Spectral Divergance algorithm
double calculateSID(std::vector<double>& refSpectrum, std::vector<double>& pixelSpectrum) {

    float sum1=0, sum2=0;
    double referenceSpecSum = 0;
    double pixelSpecSum = 0;
    for (int i=0; i<refSpectrum.size(); i++)
    {
        if (refSpectrum[i]<1)
        {
            refSpectrum[i]+=1;
        }
        if (pixelSpectrum[i]<1)
        {
            pixelSpectrum[i]+=1;
        }              
        referenceSpecSum += refSpectrum[i];
        pixelSpecSum += pixelSpectrum[i];
    }
    
    for (int i=0; i<refSpectrum.size(); i++)
    {
        double refNew = refSpectrum[i] / referenceSpecSum;
        double pixNew = pixelSpectrum[i] / pixelSpecSum;
        sum1+= (refNew)*log(refNew/pixNew);   // q_i*log(q_i/p_i)
        sum2+= (pixNew)*log(pixNew/refNew);   // p_i*log(p_i/q_i)
    }   
    
    return sum1+sum2;
    //temp_val=(sum1+sum2);
    //if (temp_val>255){temp_val=255;}


/*
    double sid_value = 0.0;
    double js_div = 0.0;

    for (size_t i = 1; i < spectrum1.size(); i++) {
        if (spectrum1[i] > 0.0 && spectrum2[i] > 0.0) {
            double m = 0.5 * (spectrum1[i] + spectrum2[i]);
            js_div += spectrum1[i] * std::log(spectrum1[i] / m);  // Use std::log
            js_div += spectrum2[i] * std::log(spectrum2[i] / m);  // Use std::log
        }
    }

    js_div *= 0.5;
    sid_value = std::sqrt(js_div);

    return sid_value;*/
}


//Calculate the Euclidean Distance
double calculateEUD(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum) {

    double sum = 0.0;
    for (size_t i = 0; i < refSpectrum.size(); ++i) {
        double diff = refSpectrum[i] - pixelSpectrum[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}

//Calculate the Spectral Correlation Mapper algorithm
double calculateSCM(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum) {
    /*std::vector<double> scmMap(hyperspectralData.size(), 0.0);

    if (targetSignature.size() != hyperspectralData[0].size()) {
        throw std::invalid_argument("Target signature dimension does not match hyperspectral data dimension.");
    }

    // Calculate the SCM for each pixel in the hyperspectral data
    for (size_t i = 1; i < hyperspectralData.size(); ++i) {
        double numerator = 0.0;
        double denominatorA = 0.0;
        double denominatorB = 0.0;

        for (size_t j = 0; j < hyperspectralData[i].size(); ++j) {
            double x = hyperspectralData[i][j];
            double y = targetSignature[j];

            numerator += x * y;
            denominatorA += x * x;
            denominatorB += y * y;
        }

        scmMap[i] = numerator / (sqrt(denominatorA) * sqrt(denominatorB));
    }

    return scmMap;*/


    double sum1=0, sum2=0, sum3=0, mean1=0, mean2=0;
    double num_layers = refSpectrum.size();
    for (int i=0; i<num_layers; i++)
    {
        mean1 += ((double)1/(double)(num_layers-1)* (double)pixelSpectrum[i]);
        mean2 += ((double)1/(double)(num_layers-1)* (double)refSpectrum[i]);
    }
    for (int i=0; i<refSpectrum.size(); i++)
    {
        sum1+=(pixelSpectrum[i]-mean1)*(refSpectrum[i]-mean2) ;
        sum2+=(pixelSpectrum[i]-mean1)*(pixelSpectrum[i]-mean1);
        sum3+=(refSpectrum[i]-mean2)*(refSpectrum[i]-mean2);
    }
    if (sum2<=0 || sum3<=0 )
    {
        return 1; // set to white due to an error
    }
    
    float temp1= sum1/(sqrt(sum2)*sqrt(sum3));
    double alpha_rad=acos(temp1);
    return (alpha_rad/3.14159);
}

//cosine similar to SAM
double calculateCOS(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum) {

    double dotProduct = 0.0;
    double sumSquaresRef = 0.0;
    double sumSquaresPixel = 0.0;

    for (size_t i = 0; i < refSpectrum.size(); ++i) {
        dotProduct += refSpectrum[i] * pixelSpectrum[i];
        sumSquaresRef += refSpectrum[i] * refSpectrum[i];
        sumSquaresPixel += pixelSpectrum[i] * pixelSpectrum[i];
    }

    double similarity = dotProduct / (sqrt(sumSquaresRef) * sqrt(sumSquaresPixel));
    return similarity;
}

//JM algorithm
double calculateJM(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum) {

    //JM relies on the vectors being probability distributions (values for each wavelength must add to 1)
    //Therefore when calculating BC we must divide the values by the integral.
    double referenceSpecSum = 0;
    double pixelSpecSum = 0;

    for (int i=0; i<refSpectrum.size(); i++)
    {
        referenceSpecSum += refSpectrum[i];
        pixelSpecSum += pixelSpectrum[i];
    }

    double BC = 0;
    for (int i=0; i<refSpectrum.size(); i++)
    {
        BC += sqrt((refSpectrum[i]/referenceSpecSum) * (pixelSpectrum[i]/pixelSpecSum));
    }

    double Bhattacharrya = -log(BC); //Intermediate step in calculating JM_distance
    double JM_distance = sqrt(2* (1 - pow(M_E, -Bhattacharrya))); //between 0 and sqrt2(2)
    double JM_distance_scaled = JM_distance / sqrt(2);
    return JM_distance_scaled; 
}


//cityblock Algorithm
double calculateCB(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum) {
 
    double distance = 0.0;

    for (size_t i = 0; i < refSpectrum.size(); ++i) {
        distance += std::abs(refSpectrum[i] - pixelSpectrum[i]);
    }

    return distance;
}