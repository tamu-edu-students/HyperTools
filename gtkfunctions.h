	#include <gtk/gtk.h>


	static void print_hello (GtkWidget *widget, gpointer data);

    static void set_pix_buf_from_cv(cv::Mat output, GtkImage *image);
    
    
    
	static void set_false_img_r(GtkSpinButton *widget,  gpointer data);
	static void set_false_img_g(GtkSpinButton *widget,  gpointer data);
	static void set_false_img_b(GtkSpinButton *widget,  gpointer data);
	static void set_min_area(GtkSpinButton *widget,  gpointer data);
	static void set_approx_poly(GtkSpinButton *widget,  gpointer data);
	static void set_img_layer(GtkSpinButton *widget,  gpointer data);	

	static void set_false_img_reset(GtkWidget *widget,  gpointer data);
	static void set_false_img_standard_rgb(GtkWidget *widget,  gpointer data);
	static void set_feature_detector_SIFT(GtkWidget *widget,  gpointer data);
	static void set_feature_detector_SURF(GtkWidget *widget,  gpointer data);
	static void set_feature_detector_ORB(GtkWidget *widget,  gpointer data);
	static void set_feature_detector_FAST(GtkWidget *widget,  gpointer data);
	static void set_feature_descriptor_SIFT(GtkWidget *widget,  gpointer data);
	static void set_feature_descriptor_SURF(GtkWidget *widget,  gpointer data);
	static void set_feature_descriptor_ORB(GtkWidget *widget,  gpointer data);
	static void set_feature_matcher_FLANN(GtkWidget *widget,  gpointer data);
	static void set_feature_matcher_BF(GtkWidget *widget,  gpointer data);	
	static void set_spec_sim_alg_SAM(GtkWidget *widget,  gpointer data);
	static void set_spec_sim_alg_SCM(GtkWidget *widget,  gpointer data);
	static void set_spec_sim_alg_SID(GtkWidget *widget,  gpointer data);

	static void show_spec_sim_img(GtkWidget *widget,  gpointer data);
	static void show_semantic_img(GtkWidget *widget,  gpointer data);
	static void show_semantic_img2(GtkWidget *widget,  gpointer data);
	static void show_contours(GtkWidget *widget,  gpointer data);
	static void show_difference(GtkWidget *widget,  gpointer data);
	static void set_spin_buttons_reset(GtkWidget *widget,  gpointer data);
	static void set_spin_buttons_standard_rgb(GtkWidget *widget,  gpointer data);
	
	static void show_false_img(GtkWidget *widget,  gpointer data);

	static void choose_image_file(GtkFileChooser *widget,  gpointer data);	

	static void print_result(GtkSpinButton *widget,  gpointer data);

	static void get_list_item(GtkComboBox *widget,  gpointer data);

	static void print_transformation(GtkWidget *widget,  gpointer data);
	static void TileImage(GtkWidget *widget,  gpointer data);
	static void feature_results(GtkWidget *widget,  gpointer data);
	static void feature_images(GtkWidget *widget,  gpointer data);	
	static void calc_spec_sim(GtkWidget *widget,  gpointer data);
	static void calc_semantic(GtkWidget *widget,  gpointer data);
	static void get_text_gtk(GtkWidget *widget,  gpointer data);
	
	
	static void load_img(GtkWidget *widget,  GtkImage*  data);
	static void clear_database(GtkWidget *widget,  gpointer data);
	static void create_database(GtkWidget *widget,  gpointer data);
	static void save_spectrum(GtkWidget *widget,  gpointer data);
	
	
	

	static void get_class_list(GtkComboBoxText *widget,  GdkEventButton *event,gpointer data);
	static void button_press_callback(GtkWidget *widget,  GdkEventButton *event, gpointer data);
	static void show_spectrum(GtkWidget *widget,GdkEventButton *event,  gpointer data);

