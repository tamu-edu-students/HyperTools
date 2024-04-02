#if !defined(HYPERFUNCTIONS_H)
#define HYPERFUNCTIONS_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <thread>

using namespace std;
// using namespace cv;

// functions that are used with the thread pool
void Classification_Child(int id, int i, cv::Mat* classified_img, cv::Mat* edge_image, vector<vector<cv::Point>>* contours_approx, vector<cv::Vec4i>* hierarchy, vector <cv::Vec3b>* contour_class);
void EdgeDetection_Child(int id, int i, cv::Mat* output_image, cv::Mat* classified_img);
//EuD stands for euclidean distance. ED was not used because of possible confusiong with edge detection
void EuD_img_Child(int id, int k, vector<cv::Mat>* mlt2, vector<vector<int>>* reference_spectrums2,cv::Mat* spec_simil_img,int* ref_spec_index);
void SAM_img_Child(int id, int k, vector<cv::Mat>* mlt1, vector<vector<int>>* reference_spectrums,cv::Mat* spec_simil_img,int* ref_spec_index);   
void SCM_img_Child(int id, int k, vector<cv::Mat>* mlt2, vector<vector<int>>* reference_spectrums2,cv::Mat* spec_simil_img,int* ref_spec_index);
void SID_img_Child(int id, int k, vector<cv::Mat>* mlt2, vector<vector<int>>* reference_spectrums2,cv::Mat* spec_simil_img,int* ref_spec_index);
void cSq_img_Child(int id, int k, vector<cv::Mat>* mlt2, vector<vector<int>>* reference_spectrums2,cv::Mat* spec_simil_img,int* ref_spec_index);
void Cos_img_Child(int id, int k, vector<cv::Mat>* mlt2, vector<vector<int>>* reference_spectrums2,cv::Mat* spec_simil_img,int* ref_spec_index);
void JM_img_Child(int id, int k, vector<cv::Mat>* mlt2, vector<vector<int>>* reference_spectrums2,cv::Mat* spec_simil_img,int* ref_spec_index);
void City_Block_Child(int id, int k, vector<cv::Mat>* mlt2, vector<vector<int>>* reference_spectrums2,cv::Mat* spec_simil_img,int* ref_spec_index);
static cv::Mat toGrayscale(cv::InputArray _src);
static cv::Mat formatImagesForPCA(const vector<cv::Mat> &data);



void similarity_img_Child(int id, int algorithmId, int columnIndex, vector<cv::Mat>& hyperspectralImage, vector<double>* reference_spectrum_ptr, cv::Mat* outputSimilarityImage);

/* /// being updated with spectralsimalgorithms.h, include file is in hyperfunctions.cpp
double CalculateSAM(const vector<double>& referenceSpectrum, const vector<double>& pixelSpectrum);
double CalculateSID(const vector<double>& referenceSpectrum, const vector<double>& pixelSpectrum);
double CalculateEuD(const vector<double>& referenceSpectrum, const vector<double>& pixelSpectrum);
double CalculatedSID(const vector<double>& referenceSpectrum, const vector<double>& pixelSpectrum);
double CalculatedCos(const vector<double>& referenceSpectrum, const vector<double>& pixelSpectrum);
double CalculatedJM(const vector<double>& referenceSpectrum, const vector<double>& pixelSpectrum);
double CalculateCityBlock(const vector<double>& referenceSpectrum, const vector<double>& pixelSpectrum);
*/

class HyperFunctions 
{

public:
	vector<cv::Mat> mlt1;  //hyperspectral image
	vector<cv::Mat> mlt2;  //hyperspectral image
	
	cv::Mat classified_img;
	cv::Mat contour_img;
	cv::Mat difference_img;
	cv::Mat edge_image;
	cv::Mat false_img;
	cv::Mat feature_img1;
	cv::Mat feature_img2;
	cv::Mat feature_img_combined; 
	cv::Mat spec_simil_img;
	cv::Mat tiled_img;
	cv::Mat pca_img;
	cv::Mat ga_img;
	cv::Mat integral_img;
	cv::Mat stitch_img;
			
	vector<string> class_list;
	vector<cv::Vec3b> color_combos;
	vector<vector<cv::Point>>contours_approx;
	vector<cv::KeyPoint> keypoints1, keypoints2;
	vector< cv::DMatch > matches;  
	vector< cv::Point2d> good_point1, good_point2;
	// vector<Vec3b> reference_colors; replaced with color_combos
	vector<vector<int>> reference_spectrums;
	
	cv::Point cur_loc=cv::Point(0, 0);
	
	double avgDist=35;
	double fov=35;
	double gps1, gps2;
	double min_area=0.0;
	double polygon_approx_coeff=0;
	double max_sim_val=0;
	double min_sim_val=255;

	bool tune_spec_sim=false;

	int classification_threshold=255; // for semantic image, if no spectra are under threshold, pixel remains black. set to 255 to classify every pixel. 15 is good to tell if pixel is of same material and allow for some noise
	int false_img_r=0; // layer value used for red channel in false image
	int false_img_g=0;  // layer value used for green channel in false image
	int false_img_b=0;  // layer value used for blue channel in false image
	int dimensionality_reduction=0;
	int feature_detector=0;
	int feature_descriptor=0;
	int feature_matcher=0;
	int num_threads=std::thread::hardware_concurrency(); // number of parallel threads
	int ref_spec_index=0;
	int spec_sim_alg=0; 
    int WINDOW_WIDTH = 800; // width of displayed image
	int WINDOW_HEIGHT= 800; // height of displayed image
    
	int filter = 0;

	string camera_database="../json/camera_database.json"; // holds camera properties
	string output_polygons="../json/file.json";  // output contour results
	string spectral_database="../json/spectral_database1.json"; // spectral and color information

	// functions to load differnt types of images
	void LoadFeatureImage1(string file_name);
	void LoadFeatureImage2(string file_name);
	void LoadImageClassified(string file_name);
	virtual void LoadImageHyper(string file_name, bool isImage1);

	//functions to display different types of images
	void DispClassifiedImage();
	void DispEdgeImage();
	void DispDifference();
	void DispContours();
	void DispTiled();
	void DispFalseImage();
	void DispSpecSim();
	void DispFeatureImgs(); 

	// various capabilities
	void DetectContours();
	void DifferenceOfImages();
	void EdgeDetection();
	void DimensionalityReduction();
	void Stitching();
	void FeatureExtraction();  
	void FeatureTransformation(); 
	void GenerateFalseImg();
	void thickEdgeContourApproximation(int idx);
	void TileImage(); // set for 164 needs to be made modular 
	void PCA_img(bool isImage1);
	
	// functions involving spectral similarity algorithms
	void EuD_img();
	void SAM_img();
	void SCM_img();
	void SemanticSegmenter();
	void SID_img();
	void Cos_img();
	void JM_img();
	void City_img();
	void SpecSimilParent();
	void cSq_img();

	//functions involving json files
	void read_img_json(string file_name);
	void read_ref_spec_json(string file_name);
	void read_spectral_json(string file_name);
	void save_new_spec_database_json();
	void save_ref_spec_json(string item_name); 
	void writeJSON(Json::Value &event, vector<vector<cv::Point> > &contours, int idx, string classification, int count);
	void writeJSON_full(vector<vector<cv::Point> > contours, vector <cv::Vec3b> contour_class,vector<cv::Vec4i> hierarchy);


	//Custom Feature Detector
	void CreateCustomFeatureDetector(int hessVal, vector<cv::KeyPoint> &keypoints, cv::Mat feature_img);
	void computeCustomDescriptor ( const cv::Mat& feature_img, std::vector<cv::KeyPoint> & keypoints,cv::Mat& descriptors);
	void gaSpace(bool isImage1);
	void ImgIntegration();

	//Spatial-Spectral SIFT detector
	void SSDetector(const cv::Mat &hyperspectralCube, std::vector<cv::KeyPoint> &keypoints);
	void SSDescriptors(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, cv::Mat &descriptor1, cv::Mat &descriptor2, float M_max);
	


	//function about match-filtering
	void filter_matches(vector<cv::DMatch> &matches);

};

#endif 