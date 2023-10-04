#if !defined(HYPERFUNCTIONS_H)
#define HYPERFUNCTIONS_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <thread>
using namespace std;
using namespace cv;

// functions that are used with the thread pool
void Classification_Child(int id, int i, Mat* classified_img, Mat* edge_image, vector<vector<Point>>* contours_approx, vector<Vec4i>* hierarchy, vector <Vec3b>* contour_class);
void EdgeDetection_Child(int id, int i, Mat* output_image, Mat* classified_img);
//EuD stands for euclidean distance. ED was not used because of possible confusiong with edge detection
void EuD_img_Child(int id, int k, vector<Mat>* mlt2, vector<vector<int>>* reference_spectrums2,Mat* spec_simil_img,int* ref_spec_index);
void SAM_img_Child(int id, int k, vector<Mat>* mlt1, vector<vector<int>>* reference_spectrums,Mat* spec_simil_img,int* ref_spec_index);   
void SCM_img_Child(int id, int k, vector<Mat>* mlt2, vector<vector<int>>* reference_spectrums2,Mat* spec_simil_img,int* ref_spec_index);
void SID_img_Child(int id, int k, vector<Mat>* mlt2, vector<vector<int>>* reference_spectrums2,Mat* spec_simil_img,int* ref_spec_index);
void Cos_img_Child(int id, int k, vector<Mat>* mlt2, vector<vector<int>>* reference_spectrums2,Mat* spec_simil_img,int* ref_spec_index);
void JM_img_Child(int id, int k, vector<Mat>* mlt2, vector<vector<int>>* reference_spectrums2,Mat* spec_simil_img,int* ref_spec_index);
void City_Block_Child(int id, int k, vector<Mat>* mlt2, vector<vector<int>>* reference_spectrums2,Mat* spec_simil_img,int* ref_spec_index);

class HyperFunctions 
{

public:
	vector<Mat> mlt1;  //hyperspectral image
	vector<Mat> mlt2;  //hyperspectral image
	
	Mat classified_img;
	Mat contour_img;
	Mat difference_img;
	Mat edge_image;
	Mat false_img;
	Mat feature_img1;
	Mat feature_img2;
	Mat feature_img_combined; 
	Mat spec_simil_img;
	Mat tiled_img;
			
	vector<string> class_list;
	vector<Vec3b> color_combos;
	vector<vector<Point>>contours_approx;
	vector<KeyPoint> keypoints1, keypoints2;
	vector< DMatch > matches;  
	vector<Vec3b> reference_colors;
	vector<vector<int>> reference_spectrums;
	
	Point cur_loc=Point(0, 0);
	
	double avgDist=35;
	double fov=35;
	double gps1, gps2;
	double min_area=0.0;
	double polygon_approx_coeff=0;

	int classification_threshold=255; // for semantic image, if no spectra are under threshold, pixel remains black. set to 255 to classify every pixel. 15 is good to tell if pixel is of same material and allow for some noise
	int false_img_r=0; // layer value used for red channel in false image
	int false_img_g=0;  // layer value used for green channel in false image
	int false_img_b=0;  // layer value used for blue channel in false image
	int feature_detector=0;
	int feature_descriptor=0;
	int feature_matcher=0;
	int num_threads=std::thread::hardware_concurrency(); // number of parallel threads
	int ref_spec_index=0;
	int spec_sim_alg=0; 
    int WINDOW_WIDTH = 800; // width of displayed image
	int WINDOW_HEIGHT= 800; // height of displayed image
    

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
	void FeatureExtraction();  
	void FeatureTransformation(); 
	void GenerateFalseImg();
	void thickEdgeContourApproximation(int idx);
	void TileImage(); // set for 164 needs to be made modular 
	
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

	//functions involving json files
	void read_img_json(string file_name);
	void read_ref_spec_json(string file_name);
	void read_spectral_json(string file_name);
	void save_new_spec_database_json();
	void save_ref_spec_json(string item_name); 
	void writeJSON(Json::Value &event, vector<vector<Point> > &contours, int idx, string classification, int count);
	void writeJSON_full(vector<vector<Point> > contours, vector <Vec3b> contour_class,vector<Vec4i> hierarchy);

};

#endif 
