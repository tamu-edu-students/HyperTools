#if !defined(GTKFUNCTIONS_H)
#define GTKFUNCTIONS_H
#include <gtk/gtk.h>
#include <iostream>
#include "gtkfunctions.h"
#include "hyperfunctions.cpp"


using namespace std;
using namespace cv;


static void print_hello (GtkWidget *widget, gpointer   data)
{
  g_print ("Hello World\n");
}


static void show_img (GtkWidget *widget, gpointer   data)
{
  g_print ("Loading Image\n");
      string file_name1="../../HyperImages/corn_fields/image_files/rgb/session_002_490_REF.jpg";
  Mat img1 = imread( file_name1, IMREAD_GRAYSCALE );
      cv::imshow("test", img1);
}

static void    TileImage(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->TileImage();
    HyperFunctions1->DispTiled();
}

static void    EdgeDetection(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->EdgeDetection();
    HyperFunctions1->DispEdgeImage();
}


static void show_semantic_img(GtkWidget *widget,  gpointer data)
{

    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->DispClassifiedImage();
}

static void show_spec_sim_img(GtkWidget *widget,  gpointer data)
{

    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->DispSpecSim();
}

static void show_contours(GtkWidget *widget,  gpointer data)
{

    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->DetectContours();
    HyperFunctions1->DispContours();
}

static void show_difference(GtkWidget *widget,  gpointer data)
{

    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->DifferenceOfImages();
    HyperFunctions1->DispDifference();
}	
	
static void print_result(GtkSpinButton *widget,  gpointer data)
{

       double result=gtk_spin_button_get_value (widget);
       cout<<result<<endl;  
}

static void set_min_area(GtkSpinButton *widget,  gpointer data)
{

    double result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->min_area=result;
    HyperFunctions1->DetectContours();
    HyperFunctions1->DispContours();
}

static void set_approx_poly(GtkSpinButton *widget,  gpointer data)
{

    double result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->polygon_approx_coeff=result;
    HyperFunctions1->DetectContours();
    HyperFunctions1->DispContours();
       
}

static void set_false_img_r(GtkSpinButton *widget,  gpointer data)
{

    int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->false_img_r=result;
    HyperFunctions1->GenerateFalseImg();
    HyperFunctions1->DispFalseImage();
       
}

static void set_false_img_g(GtkSpinButton *widget,  gpointer data)
{

    int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->false_img_g=result;
    HyperFunctions1->GenerateFalseImg();
    HyperFunctions1->DispFalseImage();
       
}

static void set_false_img_b(GtkSpinButton *widget,  gpointer data)
{

    int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->false_img_b=result;
    HyperFunctions1->GenerateFalseImg();
    HyperFunctions1->DispFalseImage();
       
}

static void set_false_img_standard_rgb(GtkWidget *widget,  gpointer data)
{


    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->false_img_r=163;
    HyperFunctions1->false_img_g=104;
    HyperFunctions1->false_img_b=65;
    HyperFunctions1->GenerateFalseImg();
    HyperFunctions1->DispFalseImage();
       
}	

static void show_false_img(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->GenerateFalseImg();
    HyperFunctions1->DispFalseImage();
}

static void feature_results(GtkWidget *widget,  gpointer data)
{


    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->FeatureExtraction();
       
}	


static void feature_images(GtkWidget *widget,  gpointer data)
{


    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->DispFeatureImgs();
       
}	

static void print_transformation(GtkWidget *widget,  gpointer data)
{


    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->FeatureExtraction();
    HyperFunctions1->FeatureTransformation();

}	

static void set_img_layer(GtkSpinButton *widget,  gpointer data)
{
    int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->feature_img1=HyperFunctions1->mlt1[result];
    HyperFunctions1->feature_img2=HyperFunctions1->mlt2[result];


}

static void set_feature_detector_SIFT(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_detector=0;}

}

static void set_feature_detector_SURF(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_detector=1;}

}
static void set_feature_detector_ORB(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_detector=2;}

}
static void set_feature_detector_FAST(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_detector=3;}

}
static void set_feature_descriptor_SIFT(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_descriptor=0;}

}
static void set_feature_descriptor_SURF(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_descriptor=1;}

}
static void set_feature_descriptor_ORB(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_descriptor=2;}

}
static void set_feature_matcher_FLANN(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_matcher=0;}

}
static void set_feature_matcher_BF(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_matcher=1;}

}

static void set_spec_sim_alg_SAM(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=0;}

}
static void set_spec_sim_alg_SCM(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=1;}

}
static void set_spec_sim_alg_SID(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=2;}

}
static void calc_spec_sim(GtkWidget *widget,  gpointer data)
{


    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->SpecSimilParent();
       
}
static void calc_semantic(GtkWidget *widget,  gpointer data)
{


    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->SemanticSegmenter();
       
}	

#endif 
