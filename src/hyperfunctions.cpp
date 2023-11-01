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

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// Loads first hyperspectral image for analysis
void HyperFunctions::LoadImageHyper1(string file_name)
{
    mlt1.clear();
	imreadmulti(file_name, mlt1);
    
}

// Loads second hyperspectral image for analysis
// mainly used for feature matching 
void HyperFunctions::LoadImageHyper2(string file_name)
{
	mlt2.clear();
	imreadmulti(file_name, mlt2);
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
void  HyperFunctions::DispFeatureImgs()
{
   Mat temp_img, temp_img2, temp_img3;
   cv::resize(feature_img1,temp_img2,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR);
   cv::resize(feature_img2,temp_img3,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR);
   Mat matArray1[]={temp_img2,temp_img3};
   hconcat(matArray1,2,temp_img);
   cv::resize(temp_img,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
   feature_img_combined=temp_img;
    //    imshow("Feature Images ", feature_img_combined);
}

// Detects, describes, and matches keypoints between 2 feature images
void  HyperFunctions::FeatureExtraction()
{
   	// feature_detector=0; 0 is sift, 1 is surf, 2 is orb, 3 is fast 
	// feature_descriptor=0; 0 is sift, 1 is surf, 2 is orb
	// feature_matcher=0; 0 is flann, 1 is bf
  //cout<<feature_detector<<" "<<feature_descriptor<<" "<<feature_matcher<<endl;
  if (feature_detector==0 && feature_descriptor==2)
  {
    cout<<"invalid detector/descriptor combination"<<endl;
    return;
  }
  
  if(feature_detector<0 || feature_detector>3 || feature_descriptor<0 || feature_descriptor>2 || feature_matcher<0 || feature_matcher>1)
  {
    cout<<"invalid feature combination"<<endl;
  }
  
  int minHessian = 400;
  Ptr<SURF> detector_SURF = SURF::create( minHessian ); 
  cv::Ptr<SIFT> detector_SIFT = SIFT::create();   
  Ptr<FastFeatureDetector> detector_FAST = FastFeatureDetector::create();
  Ptr<ORB> detector_ORB = ORB::create();
  Ptr<DescriptorMatcher> matcher;
  Mat descriptors1, descriptors2;
  
// feature_detector=0; 0 is sift, 1 is surf, 2 is orb, 3 is fast 
  if(feature_detector==0)
  {
    detector_SIFT->detect( feature_img1, keypoints1 );
    detector_SIFT->detect( feature_img2, keypoints2 );
  }
  else if (feature_detector==1)
  {
      detector_SURF->detect( feature_img1, keypoints1 );
      detector_SURF->detect( feature_img2, keypoints2 );  
  }
  else if (feature_detector==2)
  {
      detector_ORB->detect( feature_img1, keypoints1 );
      detector_ORB->detect( feature_img2, keypoints2 );  
  }
  else if (feature_detector==3)
  {
      detector_FAST->detect( feature_img1, keypoints1 );
      detector_FAST->detect( feature_img2, keypoints2 );  
  }
  	
  	// feature_descriptor=0; 0 is sift, 1 is surf, 2 is orb
  if(feature_descriptor==0)
  {
    detector_SIFT->compute( feature_img1, keypoints1 , descriptors1);
    detector_SIFT->compute( feature_img2, keypoints2 , descriptors2 );
  }  
  else if(feature_descriptor==1)
  {
    detector_SURF->compute( feature_img1, keypoints1 , descriptors1);
    detector_SURF->compute( feature_img2, keypoints2 , descriptors2 );
  }  
  else if(feature_descriptor==2)
  {
    detector_ORB->compute( feature_img1, keypoints1 , descriptors1);
    detector_ORB->compute( feature_img2, keypoints2 , descriptors2 );
  }  
  
  	// feature_matcher=0; 0 is flann, 1 is bf
  if(feature_matcher==0)
  {
    if(feature_descriptor==2) // binary descriptor 
    {
        matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        matcher->match( descriptors1, descriptors2, matches );
    }
    else
    {
        matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        matcher->match( descriptors1, descriptors2, matches );        
    }
  }    
  else if(feature_matcher==1)
  {
    if(feature_descriptor==2) // binary descriptor 
    {
        matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match( descriptors1, descriptors2, matches );
    }
    else
    {
        matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
        matcher->match( descriptors1, descriptors2, matches );        
    }  
  }   

  Mat temp_img;  
  drawMatches( feature_img1, keypoints1, feature_img2, keypoints2, matches, temp_img ); 

   cv::resize(temp_img,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
   
   feature_img_combined= temp_img;
//    imshow("Feature Images ", feature_img_combined);
}

// Finds the transformation matrix between two images
void HyperFunctions::FeatureTransformation()
{
    // camera intrinsic parameters
    //double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    
    //recovering the pose and the essential matrix
    Mat E,R,t, mask;
    vector<Point2f> points1;
    vector<Point2f> points2;
    for( int i = 0; i < matches.size(); i++ )
    {
      points1.push_back(keypoints1[matches[i].queryIdx].pt);
      points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    
    // uses ransac to filter out outliers
    E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
    // E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    //  recoverPose(E, points2, points1, R, t, focal, pp, mask);
    
    int inlier_num=0;
    for (int i=0; i<mask.rows; i++)
    {
	    int temp_val2=mask.at<uchar>(i);
	    if (temp_val2==1)
	    {
		    inlier_num+=1;
	    }
    }
    
    //cout<<" Essential Matrix"<<endl;
    //cout<<E<<endl;

    cout<<"Fundamental Matrix"<<endl;
    cout<<R<<endl<<t<<endl;

   //cout<<"inliers: " <<inlier_num<< " num of matches: "<<mask.rows<<endl;
   //cout<<" accuracy of feature matching: "<< (double)inlier_num/(double)(mask.rows)<<endl;

}

// To display classified image
void  HyperFunctions::DispClassifiedImage()
{

   Mat temp_img;
   cv::resize(classified_img,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
   imshow("Classified Image", temp_img);
}

// To display false image (RGB layers are set by the user from the hyperspectral image)
void  HyperFunctions::DispFalseImage()
{
   Mat temp_img;
   cv::resize(false_img,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
   imshow("False Image", temp_img);
}

// To display spectral similarity image
void  HyperFunctions::DispSpecSim()
{
   Mat temp_img;
   cv::resize(spec_simil_img,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
   imshow("Spectral Similarity Image", temp_img);
}

// To display edge detection image (edges are from the classified image)
void  HyperFunctions::DispEdgeImage()
{
    Mat temp_img;
    cv::resize(edge_image,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
    cv::imshow("Edge Detection Image", temp_img);
}

// Displays contour image (based on the classified image)
void  HyperFunctions::DispContours()
{
    Mat temp_img;
    cv::resize(contour_img,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
    cv::imshow("Contour Image", temp_img);
}

// Displays the differences between the classified image and the contour image
void  HyperFunctions::DispDifference()
{
    Mat temp_img;
    cv::resize(difference_img,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
    cv::imshow("Difference Image", temp_img);
}

// displays the tiled image (each image layer of the hyperspectral image is a tile)
void  HyperFunctions::DispTiled()
{
    Mat temp_img;
    cv::resize(tiled_img,temp_img,Size(WINDOW_WIDTH, WINDOW_HEIGHT),INTER_LINEAR); 
    cv::imshow("Tiled Image", temp_img);
}

// generates the false image by setting the RGB layers to what the user defines
void  HyperFunctions::GenerateFalseImg()
{

  vector<Mat> channels(3);
  channels[0]=mlt1[false_img_b]; //b
  channels[1]=mlt1[false_img_g]; //g
  channels[2]=mlt1[false_img_r]; //r
  merge(channels,false_img); // create new single channel image

    
}

//---------------------------------------------------------
// Name: DifferenceOfImages
// Description: Primarily for semantic interface tool to see how different parameters affect results. 
// Outputs a binary image with black and white pixels.
// Black pixels represents no change between the filtered/approximated image and white pixels denotes a change.
//---------------------------------------------------------
void  HyperFunctions::DifferenceOfImages()
{

    DetectContours();
	    // create a copy of the incoming image in terms of size (length and width) and initialize as an all black image
    Mat output_image(classified_img.rows, classified_img.cols, CV_8UC1, cv::Scalar(0));
    // using 8 bit image so white pixel has a value of 255

    Vec3b temp_val, compare_val; // rgb value of image at a pixel 



     for(int i = 0; i <classified_img.rows; i++) 
     {
        for(int j = 0; j < classified_img.cols; j++)
        {
            if ( classified_img.at<Vec3b>(i,j) != contour_img.at<Vec3b>(i,j))
            {
                output_image.at<uchar>(i,j)=255; 
            }
        }
     }   

    difference_img= output_image; 
	
}

// creates a binary image that sets boundary pixels as white and non-boundary pixels as black
// input is a classified image
// the output of this is used to find the contours in the image
// this is multi-threaded for speed requirements
void EdgeDetection_Child(int id, int i, Mat* output_image, Mat* classified_img2)
{

    bool edge=false;
    Vec3b temp_val, compare_val; // rgb value of image at a pixel 
    Mat classified_img=*classified_img2;
        for(int j = 0; j < classified_img.cols; j++)
        {
              edge=false;
              
              if (i==0 || j==0 ||   i== classified_img.rows-1  || j==classified_img.cols-1)
              {
                // set boundaries of image to edge
                edge=true;
              }
              else
              {
                temp_val=classified_img.at<Vec3b>(i,j); // in form (y,x) 
              
                // go through image pixel by pixel  and look at surrounding 8 pixels, if there is a difference in color between center and other pixels, then it is an edge 
                
                for (int a=-1; a<2; a++)
                {
                    for (int b=-1; b<2; b++)
                    {
                        compare_val=classified_img.at<Vec3b>(i+a,j+b);
                        if (compare_val != temp_val)
                        {
                            edge=true;
                        }            
                    }           
                }        
              }
            
              if (edge)
              {
                  // set edge pixel to white
                  output_image->at<uchar>(i,j)=255;            
              }
         }
}  
                  
void HyperFunctions::EdgeDetection( )
{
    // create a copy of the incoming image in terms of size (length and width) and initialize as an all black image
    Mat output_image(classified_img.rows, classified_img.cols, CV_8UC1, cv::Scalar(0));
    // using 8 bit image so white pixel has a value of 255

    ctpl::thread_pool p(num_threads);
    for(int i = 0; i <classified_img.rows; i++) 
    {
        p.push(EdgeDetection_Child,i,&output_image,&classified_img);
    }  

    edge_image=output_image;
}

// threaded function for DetectContours
void Classification_Child(int id, int i, Mat* classified_img, Mat* edge_image, vector<vector<Point>>* contours_approx, vector<Vec4i>* hierarchy, vector <Vec3b>* contour_class)
{
    Mat b_hist, g_hist, r_hist;
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    vector<Mat> bgr_planes;
    split( *classified_img, bgr_planes );
    Vec3b color_temp; 

     Mat drawing2 = Mat::zeros( edge_image->size(), CV_8UC1 );
     Scalar color = Scalar( 255);
     drawContours( drawing2, *contours_approx, i, color, FILLED, 8, *hierarchy, 0, Point() ); 
     calcHist( &bgr_planes[0], 1, 0, drawing2, b_hist, 1, &histSize, histRange, uniform, accumulate );
     calcHist( &bgr_planes[1], 1, 0, drawing2, g_hist, 1, &histSize, histRange, uniform, accumulate );
     calcHist( &bgr_planes[2], 1, 0, drawing2, r_hist, 1, &histSize, histRange, uniform, accumulate );
     int max_r=0, max_b=0, max_g=0;
     int max_r_loc=0, max_b_loc=0, max_g_loc=0;
     
     
        for (int j=0; j<256 ; j++)
        {
            if (r_hist.at<float>(j) > max_r)
            {
                max_r=r_hist.at<float>(j);
                max_r_loc=j;
                color_temp[1]=max_r_loc;
            }
            
            if (g_hist.at<float>(j) > max_g)
            {
                max_g=g_hist.at<float>(j);
                max_g_loc=j;
                color_temp[0]=max_g_loc;
            }
            
            if (b_hist.at<float>(j) > max_b)
            {
                max_b=b_hist.at<float>(j);
                max_b_loc=j;
                color_temp[2]=max_b_loc;
            }
      
        }
        (*contour_class)[i]=  color_temp;
}

// Description: to identify and extract the boundaries (or contours) of specific objects 
// in an image to make out the shapes of objects.
void HyperFunctions::DetectContours()
{

    //cout<<"min area "<<min_area<<" coeff poly "<<polygon_approx_coeff<<endl;
    if (edge_image.empty())
    {
    EdgeDetection();
    }

	read_spectral_json(spectral_database);

	contours_approx.clear();
    vector<Vec4i> hierarchy;
    double img_area_meters, img_area_pixels, contour_temp_area;
    img_area_pixels =  edge_image.rows*edge_image.cols ; 
    img_area_meters= pow(double(2)*avgDist* tan(fov*3.14159/double(180)/(double)2),2);   
    
    findContours( edge_image, contours_approx, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) ); 
    
    for (int i=0 ; i<contours_approx.size(); i++)
    {
        contour_temp_area = img_area_meters * contourArea(contours_approx[i]) / img_area_pixels;
        if (contour_temp_area < min_area)            
        {
            contours_approx[i].clear();                   
            contours_approx[i].push_back(Point(0, 0));
         }     
    }    
    
    Mat drawing = Mat::zeros( edge_image.size(), CV_8UC3 );
    vector <Vec3b> contour_class(contours_approx.size()+1);
    ctpl::thread_pool p(num_threads);
    for(int i = 0; i <contours_approx.size(); i++) 
    {
        if (contours_approx[i].size()>2)
        {
            p.push(Classification_Child,i,&classified_img,&edge_image, &contours_approx, &hierarchy, &contour_class);
        }
    }  
    
    // wait until threadpool is finished here
    while(p.n_idle()<num_threads)
    {
        //cout<<" running threads "<< p.size()  <<" idle threads "<<  p.n_idle()  <<endl;
        //do nothing 
    }
    
    //int count =0;
    for( int i = 0; i< contours_approx.size(); i++ )
    {
        if (contours_approx[i].size()>2)
        {
            
            Vec3b color = contour_class[i];

            string classification="unknown";
            for (int j=0; j< color_combos.size() ;j++)
            {
                if (color == color_combos[j])
                {
                    classification=class_list[j];
                }
            }
            //cout<<i<<" here "<<color<<endl;
            if (contours_approx[i][0]==Point(0,0) && contours_approx[i][1]==Point(0,edge_image.rows-1)  && contours_approx[i][2]==Point(edge_image.cols-1,edge_image.rows-1)  && contours_approx[i][3]==Point(edge_image.cols-1,0))
            {
                //writeJSON(event, contours_approx, i, "ballpark", count);
                //count++;
                Scalar temp_col=Scalar(color[2],color[0],color[1]);
                drawContours( drawing, contours_approx, i, temp_col, FILLED, 8, hierarchy, 0, Point() );
            
            }
            else
            {
                //double epsilon = polygon_approx_coeff/1000 * arcLength(contours_approx[i], true);
                // opencv method of approximating polygons
                //approxPolyDP(contours_approx[i],contour_approx_new[i],epsilon,true);
                // thick edge approximation algorithm 
                thickEdgeContourApproximation(i);
                if (contour_class[hierarchy[i][3]] != contour_class[i])
                {
                     //writeJSON(event, contours_approx, i, classification,count);
                     //count++;
                    Scalar temp_col=Scalar(color[2],color[0],color[1]);
                    drawContours( drawing, contours_approx, i, temp_col, FILLED, 8, hierarchy, 0, Point() );                
                }                           
            }               
        }
    }
    
    // uncomment to write contours to json file 
    //writeJSON_full(contours_approx, contour_class, hierarchy);
    contour_img=drawing;
    

} // end function

// Creates tile image or default/base image
// assumes 164 layers in hyperspectral image
void   HyperFunctions::TileImage()
{
    Mat empty_img= mlt1[0]*0;
    int num_chan=mlt1.size();
    int num_tile_rows=ceil(sqrt(num_chan));
    int cur_lay=0;
    vector<Mat> matArrayRows;
    Mat matArray[num_chan];
   
    for (int i=0; i<num_tile_rows; i++)
    {
        for (int j=0; j<num_tile_rows; j++)
        {
            if (cur_lay<num_chan)
            {
                matArray[j]=mlt1[cur_lay];          
            }
            else
            {
                matArray[j]=empty_img;      
            }
            cur_lay++;
        }
        
        Mat temp_row;
        hconcat(matArray,num_tile_rows,temp_row);
        matArrayRows.push_back(temp_row);
    }
    for (int i=0; i<num_tile_rows; i++)
    {
        matArray[i]=matArrayRows[i];
    }
    
    Mat temp_tile;
    vconcat(matArray,num_tile_rows,temp_tile);
   
    tiled_img=temp_tile;
    
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
void HyperFunctions::read_spectral_json(string file_name )
{

// read spectral database and return classes and rgb values 

    Vec3b color;     
 
	vector<string> class_list2; 
	color_combos.clear();
	
    ifstream ifs(file_name);
    Json::Reader reader;
    Json::Value completeJsonData;
    reader.parse(ifs,completeJsonData);
    //cout<< "Complete JSON data: "<<endl<< completeJsonData<<endl;
    
    // load rgb values and classes

    for (auto const& id3 : completeJsonData["Color_Information"].getMemberNames()) 
    {
       color[2] =  completeJsonData["Color_Information"][id3]["red_value"].asInt();
       color[0] =  completeJsonData["Color_Information"][id3]["blue_value"].asInt();
       color[1] =  completeJsonData["Color_Information"][id3]["green_value"].asInt();

      //cout<<id3<<color<<endl;
       color_combos.push_back(color);
       class_list2.push_back(id3);
    }

class_list=class_list2;
}

//---------------------------------------------------------
// Name: writeJSON
// Description: holds information about the extracted contours for navigation
//---------------------------------------------------------  
void HyperFunctions::writeJSON(Json::Value &event, vector<vector<Point> > &contours, int idx, string classification, int count)
{

    Json::Value vec(Json::arrayValue);
    for (int i = 0; i< contours[idx].size(); i++){
        Json::Value arr(Json::arrayValue);
        //cout << (contours[idx][i].x) << endl;
        //cout << (contours[idx][i].y) << endl;
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
    event["features"][count]["type"]= "Feature";
    event["features"][count]["properties"]["Name"] = Name;
    event["features"][count]["properties"]["sensor_visibility_above"] = "yes";
    event["features"][count]["properties"]["sensor_visibility_side"] = "yes";
    event["features"][count]["properties"]["traversability_av"] = "100";
    event["features"][count]["properties"]["traversability_gv"] = "100";
    event["features"][count]["properties"]["traversability_ped"] = "100";
    event["features"][count]["geometry"]["type"] = "LineString";
    event["features"][count]["geometry"]["coordinates"]=vec;
}

//---------------------------------------------------------
// Name: writeJSON
// Description: holds information about the extracted contours for navigation
//---------------------------------------------------------  
void HyperFunctions::writeJSON_full(vector<vector<Point> > contours, vector <Vec3b> contour_class,vector<Vec4i> hierarchy)
{

    std::ofstream file_id;
    file_id.open(output_polygons);
    Json::Value event; 
    // initialise JSON file 
    event["type"] = "FeatureCollection";
    event["generator"] = "Img Segmentation";    
    string Name;
     
    int count=0 ;
    int idx=0;
    string classification = class_list[idx];

    for (int idx=0; idx < contours.size()  ; idx ++)
    {    
        bool write_to_file=false;
        if (contours[idx].size()>2 && contour_class[hierarchy[idx][3]] != contour_class[idx] && idx>0)
        {            
            Vec3b color = contour_class[idx];
            classification="unknown";
            for (int j=0; j< color_combos.size() ;j++)
            {
                if (color == color_combos[j])
                {
                    classification=class_list[j];
                }
            }            

            write_to_file=true;
        }
        //else if (contours[idx][0]==Point(0,0) && contours[idx][1]==Point(0,edge_image.rows-1)  && contours[idx][2]==Point(edge_image.cols-1,edge_image.rows-1)  && contours[idx][3]==Point(edge_image.cols-1,0))
        else if (idx==0)
        {
            Name = "Ballpark";
            write_to_file=true;
        }

        if (write_to_file)
        {
            
                    
            Json::Value vec(Json::arrayValue);
            for (int i = 0; i< contours[idx].size(); i++){
                Json::Value arr(Json::arrayValue);
                arr.append(contours[idx][i].x);
                arr.append(contours[idx][i].y);
                vec.append(arr);
            }            

            if (idx>0)
            {
                Name = classification + to_string(count);
            }
            //cout<<Name<<" "<<idx<<endl;
            event["features"][count]["type"]= "Feature";
            event["features"][count]["properties"]["Name"] = Name;
            event["features"][count]["properties"]["sensor_visibility_above"] = "yes";
            event["features"][count]["properties"]["sensor_visibility_side"] = "yes";
            event["features"][count]["properties"]["traversability_av"] = "100";
            event["features"][count]["properties"]["traversability_gv"] = "100";
            event["features"][count]["properties"]["traversability_ped"] = "100";
            event["features"][count]["geometry"]["type"] = "LineString";
            event["features"][count]["geometry"]["coordinates"]=vec;
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
void  HyperFunctions::read_img_json(string file_name)
{

    ifstream ifs(file_name);
    Json::Reader reader;
    Json::Value completeJsonData;
    reader.parse(ifs,completeJsonData);

    fov=completeJsonData["FOV"].asDouble();
    avgDist=completeJsonData["AvgDistanceMeters"].asDouble();
    gps1=completeJsonData["GPS1"].asDouble();
    gps2= completeJsonData["GPS2"].asDouble();

}

// saves spectral and color information to json file of spectral curves
// assumes a ultris x20 hyperspectral image
// Accesses camera information (camera_database) and modifies spectral database
void  HyperFunctions::save_ref_spec_json(string item_name)
{
    int img_hist[mlt1.size()];
    for (int i=0; i<mlt1.size();i++)
    {
        img_hist[i]=mlt1[i].at<uchar>(cur_loc);
    }
    string user_input=item_name;
    
    // modify spectral database  
    ifstream ifs2(spectral_database);
    Json::Reader reader2;
    Json::Value completeJsonData2;
    reader2.parse(ifs2,completeJsonData2);
    
    std::ofstream file_id;
    file_id.open(spectral_database);
    Json::Value value_obj;
    value_obj = completeJsonData2;
    // save histogram to json file 
    
    for (int i=0; i<mlt1.size(); i++)
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
            return ;
        }


        value_obj["Spectral_Information"][user_input][zero_pad_result] = img_hist[i];

    }

    // change to 163, 104, 64 layer value, order is bgr

    if (mlt1.size()==164)
    {
        value_obj["Color_Information"][user_input]["red_value"]=img_hist[64];
        value_obj["Color_Information"][user_input]["blue_value"]=img_hist[104];
        value_obj["Color_Information"][user_input]["green_value"]=img_hist[163];
    }
    else 
    {
        value_obj["Color_Information"][user_input]["red_value"]=img_hist[1 * mlt1.size() / 3];
        value_obj["Color_Information"][user_input]["blue_value"]=img_hist[2 * mlt1.size() / 3];
        value_obj["Color_Information"][user_input]["green_value"]=img_hist[3 * mlt1.size() / 3-1];
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
void  HyperFunctions::read_ref_spec_json(string file_name)
{    
    // read json file 
    ifstream ifs2(file_name);
    Json::Reader reader2;
    Json::Value completeJsonData2;
    reader2.parse(ifs2,completeJsonData2);


    // initialize variables 
    int layer_values;
    Vec3b color;
    if(reference_spectrums.size()>0)
    {
    reference_spectrums.clear();
    }
    if (reference_colors.size()>0)
    {
    reference_colors.clear();
    }

    // gets spectrum of items in spectral database                         
    for (auto const& id : completeJsonData2["Spectral_Information"].getMemberNames()) 
    {
    vector<int> tempValues1;
      for (auto const& id2 : completeJsonData2["Spectral_Information"][id].getMemberNames()) 
      {
        layer_values= completeJsonData2["Spectral_Information"][id][id2].asInt();
        tempValues1.push_back(layer_values);
      }
    reference_spectrums.push_back(tempValues1);
    }

    // gets colors of items in database
    for (auto const& id3 : completeJsonData2["Color_Information"].getMemberNames()) 
    {
        color[0] =  completeJsonData2["Color_Information"][id3]["red_value"].asInt();
        color[1] =  completeJsonData2["Color_Information"][id3]["blue_value"].asInt();
        color[2] =  completeJsonData2["Color_Information"][id3]["green_value"].asInt();
        reference_colors.push_back(color);
    }
}

// creates a blank spectral database to fill in
void  HyperFunctions::save_new_spec_database_json()
{
    std::ofstream file_id3;
    file_id3.open(spectral_database);
    Json::Value new_obj;
    Json::StyledWriter styledWriter2;
    new_obj["Spectral_Information"]={};
    new_obj["Color_Information"]={};
    file_id3 << styledWriter2.write(new_obj);
    file_id3.close();

}

//---------------------------------------------------------
// Name: SemanticSegmenter
// Description: Takes hyperspectral data and assigns each pixel a color
// based on which reference spectra it is most similar to.
//---------------------------------------------------------
void  HyperFunctions::SemanticSegmenter()
{

//classified_img

    vector<Mat> temp_results;

    for (int i=0; i<reference_spectrums.size();i++)
    {
        ref_spec_index=i;
        this->SpecSimilParent();
        temp_results.push_back(spec_simil_img);
    }
    Mat temp_class_img(mlt1[1].rows, mlt1[1].cols, CV_8UC3, Scalar(0,0,0));

    for(int k = 0; k < mlt1[1].rows; k++)
    {
        for (int j=0; j < mlt1[1].cols; j++)
        {    
            double low_val;
            for (int i=0; i<temp_results.size(); i++)
            {

                if (i==0)
                {
                    low_val=temp_results[i].at<uchar>(j,k);
                    if (low_val<=classification_threshold){
                        temp_class_img.at<Vec3b>(Point(k,j)) = reference_colors[i];
                    }
                    
                }
                else
                {
                    if ( temp_results[i].at<uchar>(j,k) <low_val && temp_results[i].at<uchar>(j,k)<=classification_threshold)
                    {
                        low_val=temp_results[i].at<uchar>(j,k);
                        temp_class_img.at<Vec3b>(Point(k,j)) = reference_colors[i];
                    
                    }
                
                }
            }
        }
        
       
    }


    classified_img=temp_class_img;
}

void SpecSimilChild(int threadId, int algorithmId, int columnIndex, vector<Mat>* mlt, vector<int>* reference_spectrum_ptr, Mat* outputSimilarityImage) {

    vector<Mat> hyperspectralImage=*mlt; //dereferences
    vector<int> reference_spectrumAsInt = *reference_spectrum_ptr;
    vector<double> reference_spectrum(reference_spectrumAsInt.begin(), reference_spectrumAsInt.end());

    //Normalizes the reference vector if that is necessary for the comparison algorithm
    //Those algorithms assume that the stuff has already been normalized
    if (algorithmId == 4 || algorithmId == 6 || algorithmId == 7) {
        double reference_spectrum_sum = 0;
        for (int i = 0; i < reference_spectrum.size(); i++)
        {
            reference_spectrum_sum += reference_spectrum[i];
        }
        for (int i = 0; i < reference_spectrum.size(); i++)
        {
            reference_spectrum[i] /= reference_spectrum_sum;
        }
    }

    for (int rowIndex = 0; rowIndex <hyperspectralImage[1].rows; rowIndex++)
    {
        //Find the pixel spectrum
        vector<double> pixel_spectrum; 
        double pixel_spectrum_sum = 0;

        for (int layer = 0; layer < reference_spectrum.size(); layer++) //Assumes that pixel and reference spectra are the same size.
        {
            pixel_spectrum.push_back(hyperspectralImage[layer].at<uchar>(rowIndex,columnIndex));
            pixel_spectrum_sum += hyperspectralImage[layer].at<uchar>(rowIndex,columnIndex);
        }

        //Normalizes the pixel vector if that is necessary for the comparison algorithm
        if (algorithmId == 4 || algorithmId == 6 || algorithmId == 7) {
            for (int layer = 0; layer < reference_spectrum.size(); layer++)
            {
                pixel_spectrum[layer] /= pixel_spectrum_sum;
            }
        }

        double similarityValue = 0;

        switch(algorithmId) { //Manipulation of similarity values not complete yet...
            case 0:
                similarityValue = calculateSAM(reference_spectrum, pixel_spectrum) * 255;
                //Below is equivalent using the calculateCOS function
                //similarityValue = acos(calculateCOS(reference_spectrum, pixel_spectrum)) / 3.141592 * 255;
                break;
            case 1:
                similarityValue = calculateSCM(reference_spectrum, pixel_spectrum) * 255;
                break;
            case 2:
                similarityValue = calculateSID(reference_spectrum, pixel_spectrum) * 60;
                break;
            case 3:
                //similarityValue = calculateEUD(reference_spectrum, pixel_spectrum) / (reference_spectrum.size() + 255) * 255;
                similarityValue = calculateEUD(reference_spectrum, pixel_spectrum) / (reference_spectrum.size()) * 10;
                break;
            case 4:
                similarityValue = calculateCsq(reference_spectrum, pixel_spectrum) * 255;
                break;
            case 5:
                //calculateCOS gives high values for things that are similar, so this flips that relationship
                similarityValue = (1-calculateCOS(reference_spectrum, pixel_spectrum)) * 255;
                break;
            case 6:
                //similarityValue = (calculateCB(reference_spectrum, pixel_spectrum) / (reference_spectrum.size() + 255)) * 255;
                similarityValue = calculateCB(reference_spectrum, pixel_spectrum) / (reference_spectrum.size()) * 1000;
                break;
            case 7:
                similarityValue = calculateJM(reference_spectrum, pixel_spectrum) * 255;
                break;
            case 8: //Testing NS3
                similarityValue = sqrt(calculateEUD(reference_spectrum, pixel_spectrum)
                                      *calculateEUD(reference_spectrum, pixel_spectrum)
                                      +(1-calculateCOS(reference_spectrum, pixel_spectrum)
                                      *(1-calculateCOS(reference_spectrum, pixel_spectrum))));
        }

        outputSimilarityImage->at<uchar>(rowIndex, columnIndex) = similarityValue; 
    }
}

//---------------------------------------------------------
// Name: SpecSimilParent
// Description: to determine the similarity between sets
// of data (spectral curves) within threadpool based on their spectral properties
//---------------------------------------------------------
void  HyperFunctions::SpecSimilParent()
{
    Mat temp_img(mlt1[1].rows, mlt1[1].cols, CV_8UC1, Scalar(0));
    spec_simil_img=temp_img;

    ctpl::thread_pool p(num_threads);
    
    for (int k=0; k<mlt1[1].cols; k+=1)
    {
        p.push(SpecSimilChild, spec_sim_alg, k, &mlt1, &reference_spectrums[ref_spec_index], &spec_simil_img);
    }
}

void HyperFunctions::thickEdgeContourApproximation(int idx){
    

    int sz = contours_approx[idx].size();
    int endPt = 2;
    int stPt = 1-1;
    int midPt = 1;
    vector<vector<int>> breakPt;

    while(stPt < sz && (stPt+endPt) < sz){
        
        double xS = (double)contours_approx[idx][stPt].x;
        double yS = (double)contours_approx[idx][stPt].y;
        double xE = (double)contours_approx[idx][stPt + endPt].x;
        double yE = (double)contours_approx[idx][stPt + endPt].y;
        double xM = (double)contours_approx[idx][stPt + midPt].x;
        double yM = (double)contours_approx[idx][stPt + midPt].y;

        if(stPt == (sz-1) || (stPt + endPt) == (sz-1)  || (stPt + midPt) == (sz-1) ){
            breakPt.push_back({stPt, 0});
            break;
        }

        double num =   pow(((xM-xS)*(yE-yS)-(yM-yS)*(xE-xS)) , 2.0 );
        double den =    sqrt(((xS-xE)*(xS-xE) +  (yS-yE)*(yS-yE)));
        double dist = num/den;

        if (abs(dist) < polygon_approx_coeff){
            endPt = endPt + 1;
            midPt = midPt + 1;
        }
        else{

            breakPt.push_back({stPt, stPt + endPt - 1});
            stPt = stPt + endPt - 1;
            endPt = 2;
            midPt = 1;
        }       
    }
    vector<Point>  contour_thickEdge;

    for (int i = 0; i < breakPt.size(); i++ ){
            int idx1 = breakPt[i][0];
            int idx2 = breakPt[i][1];
            auto i1x = contours_approx[idx][idx1].x;
            auto i1y = contours_approx[idx][idx1].y;
            contour_thickEdge.push_back(Point(i1x,i1y));
    }
    int idx1 = breakPt[0][0];
    auto i1x = contours_approx[idx][idx1].x;
    auto i1y = contours_approx[idx][idx1].y;
    contour_thickEdge.push_back(Point(i1x,i1y));
    
    contours_approx[idx].clear();
    
    for (int i = 0 ; i < contour_thickEdge.size(); i++){
        contours_approx[idx].push_back(Point(contour_thickEdge[i].x,contour_thickEdge[i].y));
  
    } 

    int siz = contours_approx[idx].size();

}
