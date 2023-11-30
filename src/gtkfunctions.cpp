#if !defined(GTKFUNCTIONS_H)
#define GTKFUNCTIONS_H
#include <gtk/gtk.h>
#include <iostream>
#include "gtkfunctions.h"
#include "hyperfunctions.cpp"
#include <string>
#include <opencv2/plot.hpp>

using namespace std;
using namespace cv;

struct img_struct_gtk {
    GtkImage *image;
    HyperFunctions *HyperFunctions1;
};
  
struct entry_struct_gtk {
    GObject *entry;
    HyperFunctions *HyperFunctions1;
};

struct spin_struct_gtk {
    GtkSpinButton *button1;
    GtkSpinButton *button2;
    GtkSpinButton *button3;
    GtkSpinButton *button4;
    GtkSpinButton *button5;
    HyperFunctions *HyperFunctions1;
};
  
  
static void print_hello (GtkWidget *widget, gpointer   data)
{
    g_print ("Hello World\n");
}

static void choose_image_file(GtkFileChooser *widget,  gpointer data) {

    gchar* file_chosen;
    file_chosen = gtk_file_chooser_get_filename(widget);

    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->LoadImageHyper(file_chosen);
    
}

static void    TileImage(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
  
    HyperFunctions1->TileImage();
    cv::Mat output=HyperFunctions1->tiled_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
    set_pix_buf_from_cv( output, img_struct1->image);
    
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
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
  
    cv::Mat output=HyperFunctions1->classified_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
}

static void show_semantic_img2(GtkWidget *widget,  gpointer data)
{
     
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->DispClassifiedImage();

}

static void show_spec_sim_img(GtkWidget *widget,  gpointer data)
{   
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
  
    cv::Mat output=HyperFunctions1->spec_simil_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
    
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
static void set_zoom1(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->polygon_approx_coeff=10;
    HyperFunctions1->min_area=0.2;
    HyperFunctions1->DetectContours();
    HyperFunctions1->DispContours();    
}

static void set_zoom2(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->polygon_approx_coeff=100;
    HyperFunctions1->min_area=1;
    HyperFunctions1->DetectContours();
    HyperFunctions1->DispContours();    
}

static void set_zoom3(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->polygon_approx_coeff=1000;
    HyperFunctions1->min_area=2;
    HyperFunctions1->DetectContours();
    HyperFunctions1->DispContours();    
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
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    HyperFunctions1->false_img_r=result;
    HyperFunctions1->GenerateFalseImg();

    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
       
}

static void set_false_img_g(GtkSpinButton *widget,  gpointer data)
{
    int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    HyperFunctions1->false_img_g=result;
    HyperFunctions1->GenerateFalseImg();

    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
    
    set_pix_buf_from_cv( output, img_struct1->image);
       
}

static void set_false_img_b(GtkSpinButton *widget,  gpointer data)
{
    
    int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    HyperFunctions1->false_img_b=result;
    HyperFunctions1->GenerateFalseImg();

    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
       
}
static void set_image_width(GtkSpinButton *widget,  gpointer data)
{  
     int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    HyperFunctions1->WINDOW_WIDTH=result;
    HyperFunctions1->GenerateFalseImg();
    

    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);


}
static void set_image_height(GtkSpinButton *widget,  gpointer data)
{
     int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    HyperFunctions1->WINDOW_HEIGHT=result;
    HyperFunctions1->GenerateFalseImg();
    

    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
}

static void set_pix_buf_from_cv(cv::Mat output, GtkImage *image)
{
 
    int channels=output.channels();
    GdkPixbuf* pixbuf=gdk_pixbuf_new(GDK_COLORSPACE_RGB, channels==4, 8, output.cols, output.rows);
    guchar* data3=gdk_pixbuf_get_pixels(pixbuf);
    int stride = gdk_pixbuf_get_rowstride(pixbuf);
    for (int y=0;y<output.rows;y++)
    {
        for(int x=0;x<output.cols;x++)
        {
            if (channels==3)
            {
                cv::Vec3b pixel = output.at<cv::Vec3b>(y,x);
                data3[y*stride+x*channels+ 0]=pixel[2];
                data3[y*stride+x*channels+ 1]=pixel[1];
                data3[y*stride+x*channels+ 2]=pixel[0];    
        
            }
            else if(channels==4)
            {
                data3[y*stride+x*channels+ 3]=255;  
            }
            else if(channels==1)
            {
                int pixel = output.at<uchar>(y,x);
                data3[y*stride+3*x*channels+ 0]=pixel;
                data3[y*stride+3*x*channels+ 1]=pixel;
                data3[y*stride+3*x*channels+ 2]=pixel;   
            }        
        } 
    }
    gtk_image_set_from_pixbuf(image, pixbuf);
}

static void set_false_img_reset(GtkWidget *widget,  gpointer data) {
     
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
  
    HyperFunctions1->false_img_r=0;
    HyperFunctions1->false_img_g=0;
    HyperFunctions1->false_img_b=0;
    HyperFunctions1->GenerateFalseImg();
 
    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
}

static void set_false_img_standard_rgb(GtkWidget *widget,  gpointer data)
{

    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);  
    
    //query size of mlt1, find out how many layers in the standard image
    // std::cout << HyperFunctions1->mlt1.size() << std::endl;

    if(HyperFunctions1->mlt1.size() == 51){ //ultris5
        HyperFunctions1->false_img_r=31; //Hard-coded for the ultris5 camera
        HyperFunctions1->false_img_g=13;
        HyperFunctions1->false_img_b=2;
    }
    else if(HyperFunctions1->mlt1.size() == 164){
        HyperFunctions1->false_img_r=163;
        HyperFunctions1->false_img_g=104;
        HyperFunctions1->false_img_b=65;
    }
    else{
        int size = HyperFunctions1->mlt1.size();
        HyperFunctions1->false_img_r=size-1;
        HyperFunctions1->false_img_g=size*2/3;
        HyperFunctions1->false_img_b=size/3;
    }
    HyperFunctions1->GenerateFalseImg();
 
    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
}

static void set_spin_buttons_reset(GtkWidget *widget,  gpointer data) 
{

    void * data_new=data;
    spin_struct_gtk *spin_struct1=static_cast<spin_struct_gtk*>(data_new);
    //void * data_new2=spin_struct1->HyperFunctions1;
    //HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);

    gtk_spin_button_set_value((*spin_struct1).button1,0);
    gtk_spin_button_set_value((*spin_struct1).button2,0);
    gtk_spin_button_set_value((*spin_struct1).button3,0);
}

static void adjust_spin_ranges(GtkWidget *widget,  gpointer data) {
    void * data_new=data;
    spin_struct_gtk *spin_struct1=static_cast<spin_struct_gtk*>(data_new);
    void * data_new2=spin_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);

    std::cout << HyperFunctions1->mlt1.size()-1 << std::endl;
    gtk_spin_button_set_range((*spin_struct1).button1, 0, HyperFunctions1->mlt1.size());
    gtk_spin_button_set_range((*spin_struct1).button2, 0, HyperFunctions1->mlt1.size());
    gtk_spin_button_set_range((*spin_struct1).button3, 0, HyperFunctions1->mlt1.size());
}

static void set_spin_buttons_standard_rgb(GtkWidget *widget,  gpointer data) {
    void * data_new=data;
    spin_struct_gtk *spin_struct1=static_cast<spin_struct_gtk*>(data_new);
    void * data_new2=spin_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);

    gtk_spin_button_set_value((*spin_struct1).button1,HyperFunctions1->false_img_r);
    gtk_spin_button_set_value((*spin_struct1).button2,HyperFunctions1->false_img_g);
    gtk_spin_button_set_value((*spin_struct1).button3,HyperFunctions1->false_img_b);
}

static void show_false_img(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    HyperFunctions1->GenerateFalseImg();
    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 

    set_pix_buf_from_cv( output, img_struct1->image);
}

static void show_ndvi_image(GtkWidget *widget,  gpointer data)
{
    //Along with the set_false_img_standard_rgb function, this uses bands that are hardcoded.
    //To ensure compatibility with other cameras this should be changed in the future.
    
    // not currently functional, this is a work in progress

    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    //HyperFunctions1->GenerateFalseImg();
    //NDVI= (NIR-R) / (NIR +R)
    //All channels will display the same formula until a color version is discovered
    vector<Mat>& mlt1 = HyperFunctions1->mlt1;
    Mat ndvi_img;
    Mat white_img(mlt1[1].rows, mlt1[1].cols, CV_8UC1, Scalar(255,255,255));  
    vector<Mat> channels(3);
    int r = 163;
    int nir = 112; //In progress. I think the problem is that the values are unsigned so hitting 0 keeps it there.
    channels[0]=(((mlt1[nir] - mlt1[r]) / (mlt1[nir] + mlt1[r]))*100000)+white_img/2; //b
    channels[1]=channels[0]; //g
    channels[2]=channels[0]; //b
    merge(channels,ndvi_img); // create new single channel image

    cv::Mat output=ndvi_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
    
    set_pix_buf_from_cv( output, img_struct1->image);
}

static void clear_database(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->save_new_spec_database_json();

}

static void create_database(GtkWidget *widget,  gpointer data)
{

    void * data_new=data;
    entry_struct_gtk *entry_struct1=static_cast<entry_struct_gtk*>(data_new);
    void * data_new2=entry_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    string text_val=gtk_entry_get_text(GTK_ENTRY(entry_struct1->entry));
    HyperFunctions1->spectral_database="../json/" + text_val;
    HyperFunctions1->save_new_spec_database_json();
}

static void choose_database(GtkFileChooser *widget,  gpointer data)
{

    gchar* file_chosen;
    file_chosen = gtk_file_chooser_get_filename(widget);
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->spectral_database=file_chosen;                 //Changes variable in spectral database
}
static void save_spectrum(GtkWidget *widget,  gpointer data)
{
    
    void * data_new=data;
    entry_struct_gtk *entry_struct1=static_cast<entry_struct_gtk*>(data_new);
    void * data_new2=entry_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    string text_val=gtk_entry_get_text(GTK_ENTRY(entry_struct1->entry));
    HyperFunctions1->save_ref_spec_json(text_val.c_str());
  
}

static void feature_results(GtkWidget *widget,  gpointer data)
{
    // void * data_new=data;
    // HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    // HyperFunctions1->FeatureExtraction();


    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
  
    HyperFunctions1->FeatureExtraction();
   
 
    cv::Mat output=HyperFunctions1->feature_img_combined;

    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);

/*
 int result=gtk_spin_button_get_value (widget);
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
    HyperFunctions1->false_img_r=result;
    HyperFunctions1->GenerateFalseImg();

    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);

*/


}	

static void feature_images(GtkWidget *widget,  gpointer data)
{
    // void * data_new=data;
    // HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    // HyperFunctions1->DispFeatureImgs();

    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
  
    HyperFunctions1->DispFeatureImgs();
   
 
    cv::Mat output=HyperFunctions1->feature_img_combined;

    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
       
}	

static void print_transformation(GtkWidget *widget,  gpointer data)
{

    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->FeatureExtraction();
    HyperFunctions1->FeatureTransformation();

}	

static void get_text_gtk(GtkWidget *widget,  gpointer data)
{
    //void * data_new=data;
    //HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    cout<<gtk_entry_get_text(GTK_ENTRY(widget))<<endl;
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
static void set_filter_1(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->filter=1;}

}
static void set_filter_na(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->filter=0;}

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
static void set_feature_detector_SSSift(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_detector=4;}

}
static void set_feature_detector_example_detector(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_detector=5;}

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
static void set_feature_descriptor_SSSift(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_descriptor=3;}

}
static void set_feature_descriptor_example_detector(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->feature_descriptor=4;}

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

static void set_spec_sim_alg_EuD(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=3;}
}

static void set_spec_sim_alg_chi_squared(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=4;}
}
static void set_spec_sim_alg_cosine_similarity(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=5;}
}

static void set_spec_sim_alg_city_block(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=6;}
}

static void set_spec_sim_alg_jm_distance(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=7;}
}

static void set_spec_sim_alg_NS3(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=8;}
}

static void set_spec_sim_alg_JM_SAM(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=9;}
}

static void set_spec_sim_alg_SCA(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=10;}
}

static void set_spec_sim_alg_SID_SAM(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=11;}
}

static void set_spec_sim_alg_SID_SCA(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=12;}
}

static void set_spec_sim_alg_hellinger(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=13;}
}

static void set_spec_sim_alg_canberra(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) {HyperFunctions1->spec_sim_alg=14;}
}

static void load_img(GtkWidget *widget,  GtkImage*  data)
{
  
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);
  
    cv::Mat output=HyperFunctions1->false_img;
    cv::resize(output,output,Size(HyperFunctions1->WINDOW_WIDTH, HyperFunctions1->WINDOW_HEIGHT),INTER_LINEAR); 
    set_pix_buf_from_cv( output, img_struct1->image);
}


static void button_press_callback(GtkWidget *widget,  GdkEventButton *event, gpointer data)
{
    
    // g_print ("Event box clicked at coordinates %f,%f\n",event->x, event->y);
             
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    
    double click_x, click_y, img_x, img_y;
    click_x = (event->x);
    click_y = (event->y);
    img_x = (click_x/ double(HyperFunctions1->WINDOW_WIDTH)  * double(HyperFunctions1->mlt1[0].cols));
    img_y = (click_y/ double(HyperFunctions1->WINDOW_HEIGHT)  * double(HyperFunctions1->mlt1[0].rows));
    HyperFunctions1->cur_loc=Point(img_x, img_y );         
    cout<<click_x<<" , "<<click_y<<" convert "<<img_x<<" , "<<img_y<<endl;
             
}

static void show_spectrum(GtkWidget *widget, GdkEventButton *event, gpointer data)
{
    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);

    // show the spectrum here 
    // https://github.com/opencv/opencv_contrib/blob/master/modules/plot/samples/plot_demo.cpp 
    // https://docs.opencv.org/4.x/d0/d1e/classcv_1_1plot_1_1Plot2d.html
    
    Mat data_x( 1, HyperFunctions1->mlt1.size(), CV_64F ); // wavelength
    Mat data_y( 1, HyperFunctions1->mlt1.size(), CV_64F ); // reflectance value

    for ( int i = 0; i < data_x.cols; i++ )
    {
        data_x.at<double>( 0, i ) = i;
        data_y.at<double>( 0, i ) = HyperFunctions1->mlt1[i].at<uchar>(HyperFunctions1->cur_loc);
    }

    Mat plot_result;
    Ptr<plot::Plot2d> plot = plot::Plot2d::create( data_x, data_y );
    plot->render(plot_result);
    plot->setShowText( false );
    plot->setPlotBackgroundColor( Scalar( 255, 200, 200 ) );
    plot->setPlotLineColor( Scalar( 255, 0, 0 ) );
    plot->setPlotLineWidth( 2 );
    plot->setInvertOrientation( true );
    plot->setMinY(0);
    plot->setMaxY(256);
        
    plot->render(plot_result);
    cv::Mat output=plot_result;        //HyperFunctions1->false_img;
    cv::resize(output,output,Size(400,200),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);
    //imshow("Test", output);
    //cout << "THis is running" << endl;
}

static void button_callback_and_show_spectrum(GtkWidget *widget, GdkEventButton *event, gpointer data)
{
    //Button ballback part
    //void * data_new=data;
    //HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);

    void * data_new=data;
    img_struct_gtk *img_struct1=static_cast<img_struct_gtk*>(data_new);
    void * data_new2=img_struct1->HyperFunctions1;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new2);

    double click_x, click_y, img_x, img_y;
    click_x = (event->x);
    click_y = (event->y);
    img_x = (click_x/ double(HyperFunctions1->WINDOW_WIDTH)  * double(HyperFunctions1->mlt1[0].cols));
    img_y = (click_y/ double(HyperFunctions1->WINDOW_HEIGHT)  * double(HyperFunctions1->mlt1[0].rows));
    HyperFunctions1->cur_loc=Point(img_x, img_y );         
    

    // Show spectrum Part
    

    // show the spectrum here 
    // https://github.com/opencv/opencv_contrib/blob/master/modules/plot/samples/plot_demo.cpp 
    // https://docs.opencv.org/4.x/d0/d1e/classcv_1_1plot_1_1Plot2d.html
    
    Mat data_x( 1, HyperFunctions1->mlt1.size(), CV_64F ); // wavelength
    Mat data_y( 1, HyperFunctions1->mlt1.size(), CV_64F ); // reflectance value


    for ( int i = 0; i < data_x.cols; i++ )
    {
        data_x.at<double>( 0, i ) = i;
        data_y.at<double>( 0, i ) = HyperFunctions1->mlt1[i].at<uchar>(HyperFunctions1->cur_loc);
    }

    Mat plot_result;
    Ptr<plot::Plot2d> plot = plot::Plot2d::create( data_x, data_y );
    plot->render(plot_result);
    plot->setShowText( false );
    plot->setPlotBackgroundColor( Scalar( 255, 200, 200 ) );
    plot->setPlotLineColor( Scalar( 255, 0, 0 ) );
    plot->setPlotLineWidth( 2 );
    plot->setInvertOrientation( true );
    plot->setMinY(0);
    plot->setMaxY(256);
    
    plot->render(plot_result);
    cv::Mat output=plot_result;        //HyperFunctions1->false_img;
    cv::resize(output,output,Size(400,200),INTER_LINEAR); 
  
    set_pix_buf_from_cv( output, img_struct1->image);

}

static void calc_spec_sim(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->read_ref_spec_json(HyperFunctions1->spectral_database);
    HyperFunctions1->SpecSimilParent();

}
static void calc_semantic(GtkWidget *widget,  gpointer data)
{
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->read_ref_spec_json( HyperFunctions1->spectral_database);
    HyperFunctions1->SemanticSegmenter();

}	


static void get_class_list(GtkComboBoxText *widget, GdkEventButton *event, gpointer data)
{
    if (data!=NULL)
    {
        void * data_new=data;
        HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
        HyperFunctions1->read_spectral_json(HyperFunctions1 -> spectral_database);

        // import class list here and add to textbox
        vector<string> class_list= HyperFunctions1->class_list;
        
        gtk_combo_box_text_remove_all(widget);
        for (int i=0; i<class_list.size(); i++)
        {
            string temp1=class_list[i];
            const char  *temp_char2=const_cast<char*>(temp1.c_str());
            const char *temp_char1=std::to_string(i).c_str();
            gtk_combo_box_text_insert (widget, 0, temp_char1,temp_char2);
        }
    }
}

static void get_list_item(GtkComboBox *widget,  gpointer data)
{

    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->read_spectral_json( HyperFunctions1 -> spectral_database);
    const gchar* active_text=gtk_combo_box_get_active_id(widget);
    if (active_text!=NULL)
    {
        string list_index=static_cast<string>(active_text);
        //cout<<list_index<<endl; // prints index of selected value bottom to the top
        HyperFunctions1->ref_spec_index=stoi(list_index);
    }

}


#endif 
