#include <cmath>     // Include cmath for mathematical functions
#include <iostream>
#include <vector>    // Include vector for std::vector


//Calculate the Spectral Angle Mapper algorithm
double CalculateSAM(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2) {
    if (spectrum1.size() != spectrum2.size()) {
        throw std::invalid_argument("Number of spectral bands do not match.");
    }
    // Calculate the dot product of the two spectral vectors
    double dotProduct = 0.0;
    double magnitude1 = 0.0;
    double magnitude2 = 0.0;

    for (size_t i = 0; i < spectrum1.size(); i++) {
        dotProduct += spectrum1[i] * spectrum2[i];
        magnitude1 += spectrum1[i] * spectrum1[i];
        magnitude2 += spectrum2[i] * spectrum2[i];
    }

    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);

    // Calculate the spectral angle
    double cosineSimilarity = dotProduct / (magnitude1 * magnitude2);
    double angle = std::acos(cosineSimilarity);  // Angle in radians

    return angle;
}
//Calculate the Spectral Divergance algorithm
double calculateSID(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2) {

    if (spectrum1.size() != spectrum2.size()) {
        throw std::invalid_argument("Number of spectral bands do not match.");
    }


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

    return sid_value;
}


//Calculate the Euclidean Distance
double calculateEUD(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2) {
    if (spectrum1.size() != spectrum2.size()) {
        throw std::invalid_argument("Number of spectral bands do not match.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < spectrum1.size(); ++i) {
        double diff = spectrum1[i] - spectrum2[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

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
double calculateCOS(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2) {
    if (spectrum1.size() != spectrum2.size()) {
        throw std::invalid_argument("Spectra must have the same length.");
    }

    double dotProduct = 0.0;
    double normSpectrum1 = 0.0;
    double normSpectrum2 = 0.0;

    for (size_t i = 0; i < spectrum1.size(); ++i) {
        dotProduct += spectrum1[i] * spectrum2[i];
        normSpectrum1 += spectrum1[i] * spectrum1[i];
        normSpectrum2 += spectrum2[i] * spectrum2[i];
    }

    if (normSpectrum1 == 0.0 || normSpectrum2 == 0.0) {
        throw std::invalid_argument("One of the spectra has zero norm.");
    }

    double similarity = dotProduct / (sqrt(normSpectrum1) * sqrt(normSpectrum2));
    return similarity;
}

//JM algorithm
double calculateJM(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2) {
    double jmDistance = 0.0;
    for (size_t i = 1; i < spectrum1.size(); ++i) {
        double sqrtProduct = sqrt(spectrum1[i] * spectrum2[i]);
        jmDistance += sqrtProduct;
    }

    jmDistance = 2.0 - 2.0 * jmDistance;

    return sqrt(jmDistance);
}


//cityblock Algorithm
double calculateCB(const std::vector<double>& spectrum1, const std::vector<double>& spectrum2) {
 
    double distance = 0.0;
    for (size_t i = 0; i < spectrum1.size(); ++i) {
        distance += std::abs(spectrum1[i] - spectrum2[i]);
    }

    return distance;
}