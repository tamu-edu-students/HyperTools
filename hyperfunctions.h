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
void SAM_img_Child(int id, int k, vector<Mat>* mlt1, vector<vector<int>>* reference_spectrums,Mat* spec_simil_img,int* ref_spec_index);   
void SID_img_Child(int id, int k, vector<Mat>* mlt2, vector<vector<int>>* reference_spectrums2,Mat* spec_simil_img,int* ref_spec_index);
void SCM_img_Child(int id, int k, vector<Mat>* mlt2, vector<vector<int>>* reference_spectrums2,Mat* spec_simil_img,int* ref_spec_index);

class HyperFunctions 
{

public:
	vector<Mat> mlt1;  //hyperspectral image
	vector<Mat> mlt2;  //hyperspectral image
	Mat classified_img;
	Mat edge_image;
	Mat contour_img;
	Mat difference_img;
	Mat tiled_img;
	Mat feature_img1;
	Mat feature_img2;
	Mat false_img;
	Mat spec_simil_img;
		
	vector<string> class_list;
	vector<Vec3b> color_combos;
	vector< DMatch > matches;  
	vector<KeyPoint> keypoints1, keypoints2;
	vector<vector<int>> reference_spectrums;
	vector<Vec3b> reference_colors;
	
	Point cur_loc=Point(0, 0);

	double polygon_approx_coeff=0;
	double avgDist=35;
	double fov=35;
	double min_area=0.0;
	double gps1, gps2;

	int WINDOW_WIDTH =1000;
	int WINDOW_HEIGHT= 800;
	int feature_detector=0;
	int feature_descriptor=0;
	int feature_matcher=0;
	int false_img_r=0;
	int false_img_g=0;
	int false_img_b=0;
	int spec_sim_alg=0;
	int ref_spec_index=0;
	int num_threads=std::thread::hardware_concurrency();

	string spectral_database="../json/spectral_database1.json";
	string camera_database="../json/camera_database.json";
	string output_polygons="../json/file.json";

	// functions to load differnt types of images
	void LoadImageHyper1(string file_name);
	void LoadImageHyper2(string file_name);
	void LoadImageClassified(string file_name);
	void LoadFeatureImage1(string file_name);
	void LoadFeatureImage2(string file_name);

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
	void GenerateFalseImg();
	void TileImage(); // set for 164 needs to be made modular 
	void FeatureExtraction();  
	void FeatureTransformation(); 
	
	// functions involving spectral similarity algorithms
	void SAM_img();
	void SID_img();
	void SCM_img();
	void SemanticSegmenter();
	void SpecSimilParent();

	//functions involving json files
	void read_spectral_json(string file_name);
	void writeJSON(Json::Value &event, vector<vector<Point> > &contours, int idx, string classification, int count);
	void read_img_json(string file_name);
	void save_ref_spec_json(string file_name); // save real spectrum
	void read_ref_spec_json(string file_name);
	void save_new_spec_database_json(string file_name);

};

#endif 
