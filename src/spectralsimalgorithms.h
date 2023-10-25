#pragma once
#include <cmath>     // Include cmath for mathematical functions
#include <iostream>
#include <vector>    // Include vector for std::vector

//Calculate the Spectral Angle Mapper algorithm
double calculateSAM(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum);

//Calculate the Spectral Divergance algorithm
double calculateSID(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum);


//Calculate the Euclidean Distance
double calculateEUD(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum);

//Calculate the Spectral Correlation Mapper algorithm
double calculateSCM(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum);

//cosine similar to SAM
double calculateCOS(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum);

//JM algorithm
double calculateJM(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum);

//cityblock Algorithm
double calculateCB(const std::vector<double>& refSpectrum, const std::vector<double>& pixelSpectrum);