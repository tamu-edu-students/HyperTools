#if !defined(GTKFUNCTIONS_H)
#define GTKFUNCTIONS_H
#include <gtk/gtk.h>
#include <iostream>
#include "gtkfunctions.h"
#include "hyperfunctions.cpp"
#include <string>

using namespace std;
using namespace cv;

static void print_hello (GtkWidget *widget, gpointer   data)
{
  g_print ("Hello World\n");
}

static void choose_image_file(GtkFileChooser *widget,  gpointer data) {

    gchar* file_chosen;
    file_chosen = gtk_file_chooser_get_filename(widget);
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->LoadImageHyper1(file_chosen);
    
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

static void load_img(GtkWidget *widget,  GtkImage*  data)
{

  //  gtk_image_clear (data);  GtkImage*
  gtk_image_set_from_file(data,"../lena3.png");
  g_print ("Hello World\n");
}


static void button_press_callback(GtkWidget *widget,  GdkEventButton *event, gpointer data)
{
    
 g_print ("Event box clicked at coordinates %f,%f\n",
             event->x, event->y);
}



static void calc_spec_sim(GtkWidget *widget,  gpointer data)
{


    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) 
    {
    HyperFunctions1->read_ref_spec_json( "../json/spectral_database1.json");
    HyperFunctions1->SpecSimilParent();
    }

       
}
static void calc_semantic(GtkWidget *widget,  gpointer data)
{


    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    gboolean T = gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget));
    if (T==1) 
    {
        HyperFunctions1->read_ref_spec_json( "../json/spectral_database1.json");
        HyperFunctions1->SemanticSegmenter();
    }
       
}	


static void get_class_list(GtkComboBoxText *widget, GdkEventButton *event, gpointer data)
{

    if (data!=NULL)
    {
    void * data_new=data;
        HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
        HyperFunctions1->read_spectral_json( "../json/spectral_database1.json");

        // import class list here and add to textbox
        vector<string> class_list= HyperFunctions1->class_list;
        
        //cout<<"fov"<<HyperFunctions1->fov<<endl;
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

        //cout<<data<<endl;
    void * data_new=data;
    HyperFunctions *HyperFunctions1=static_cast<HyperFunctions*>(data_new);
    HyperFunctions1->read_spectral_json( "../json/spectral_database1.json");
    const gchar* active_text=gtk_combo_box_get_active_id(widget);
    if (active_text!=NULL)
    {
    string list_index=static_cast<string>(active_text);
    //cout<<list_index<<endl; // prints index of selected value bottom to the top
    HyperFunctions1->ref_spec_index=stoi(list_index);
    }

}

#endif 
