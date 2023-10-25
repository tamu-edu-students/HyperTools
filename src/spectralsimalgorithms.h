#include <cmath>     // Include cmath for mathematical functions
#include <iostream>
#include <vector>    // Include vector for std::vector

//Calculate the Spectral Angle Mapper algorithm
double CalculateSAM(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2);

//Calculate the Spectral Divergance algorithm
double calculateSID(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2);


//Calculate the Euclidean Distance
double calculateEUD(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2);

//Calculate the Spectral Correlation Mapper algorithm
std::vector<double> calculateSCM(const std::vector<std::vector<double>>& hyperspectralData, const std::vector<double>& targetSignature) {
    std::vector<double> scmMap(hyperspectralData.size(), 0.0);

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

    return scmMap;
}

//cosine similar to SAM
double calculateCOS(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2);

//JM algorithm
double calculateJM(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2);


//cityblock Algorithm
double calculateCB(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2);