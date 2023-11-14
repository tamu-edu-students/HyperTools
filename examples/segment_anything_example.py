import sys
sys.path.insert(2,'submodules/segment-anything/')
sys.path.append("submodules/segment-anything/segment_anything/")


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import time 
from matplotlib import animation
import json 
from multiprocessing import cpu_count, Process, Value, Array, Pool, TimeoutError
import math
#using opencl instead of cuda for improved platform support
from scipy import stats as st
from pyopencl.tools import get_test_platforms_and_devices
import pyopencl as cl

#problem with cuvis code, need to fix
# import cuvis
# lib_dir = os.getenv("CUVIS")
# data_dir = os.path.normpath(os.path.join(lib_dir, os.path.pardir, "sdk", "sample_data", "set1"))

 # Define the OpenCL kernel code

    
def extract_rgb(cube, red_layer=78 , green_layer=40, blue_layer=25,  visualize=False):

    
    try: #for loading using opencv
        red_img = cube[:,:, red_layer]
        green_img = cube[:,:, green_layer]
        blue_img = cube[:,:, blue_layer]
    except: #for loading using cuvis
        red_img=cube.array[:,:, red_layer]
        green_img=cube.array[:,:, green_layer]
        blue_img=cube.array[:,:, blue_layer]
        
    data=np.stack([red_img,green_img,blue_img], axis=-1)
    #print(image.shape)
    #print(type(image))

    #convert to 8bit
    x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
    image=(x_norm*255).astype('uint8')
    if visualize:
        #pass
        plt.imshow(image)
        plt.show()
    return image  

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def segment_img(sam,rgb_img):
    
    #generate masks on rgb image using the base parameters
    # startTime = time.time()
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(rgb_img)
    # endTime = time.time()
    # print("time to generate masks: ", endTime-startTime)
    return masks
    
    
    # # tune SAM parameters to generate more masks
    # # parameters were found online and havent been tuned 
    # # startTime = time.time()
    # mask_generator_2 = SamAutomaticMaskGenerator(
    # model=sam,
    # points_per_side=32,
    # pred_iou_thresh=0.86,
    # stability_score_thresh=0.92,
    # crop_n_layers=1,
    # crop_n_points_downscale_factor=2,
    # min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )
    # masks2 = mask_generator_2.generate(rgb_img)
    # # endTime = time.time()
    # # print("time to generate masks: ", endTime-startTime)
    # return masks2
    
    #documentation for masks
    #masks is a list of dictionaries
    #keys of dictionary are segmentation, area, bbox, predicted_iou, point_coords, stability_score, crop_box
    #segmentation - np array of size equal to input image , values true and false
    #"segmentation"          : dict,             # Mask saved in COCO RLE format.
    # print(len(list(zip(*np.where(masks[0]['segmentation'] == True)))))
    # area is a value equal to length of segmentation aka num of pixels
    # "area"                  : int,              # The area in pixels of the mask
    # print(masks[0]['area'])
    #bbox has 4 values , x,y,w,h
    #"bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    #print(masks[0]['bbox'])
    #predicted_iou is a value, intersection over union, higher is better
    #"predicted_iou"         : float,            # The model's own prediction of the mask's quality
    #print(masks[0]['predicted_iou'])
    #print(masks[0]['stability_score'])
    #"stability_score"       : float,            # A measure of the mask's quality
    #higher is better
    #print(masks[0]['crop_box'])
    #"crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    #if it is the whole image than it will be 0,0,img width, img height
    #print(masks[0]['point_coords'])
    #"point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
    #pixel coordinates that were inputted to the model to generate the mask 

def semantic_class(data):

    arr=np.zeros((data.shape[0],data.shape[1]))
    
    for i in range((data.shape[0])):
        for j in range((data.shape[1])):
            #print(data[i,j,:], np.argmin(data[i,j,:]))
            arr[i,j]=np.argmin(data[i,j,:])
    return arr

def read_spec_json(file_name):
    
    f=open(file_name)
    data=json.load(f)
    
    color_array=np.array([])
    spectral_array=np.array([])
    for i in data['Color_Information']:
        temp_array=np.array([])
        for j in data['Color_Information'][i]:
            #print(i,j,data['Color_Information'][i][j])
            temp_array=np.hstack((temp_array, np.array([data['Color_Information'][i][j]]))) 
        if (len(color_array)==0):
            color_array =np.hstack((color_array,temp_array))
        else:
            color_array =np.vstack((color_array,temp_array))
    for i in data['Spectral_Information']:
        temp_array=np.array([])
        for j in data['Spectral_Information'][i]:
            #print(i,j,data['Spectral_Information'][i][j])
            temp_array=np.hstack((temp_array, np.array([data['Spectral_Information'][i][j]])))
        if (len(spectral_array)==0):
            spectral_array =np.hstack((spectral_array,temp_array))
        else:
            spectral_array =np.vstack((spectral_array,temp_array))
               
    f.close()
    return color_array, spectral_array   
    
def semantic_color(data,color_array):

    arr=np.zeros((data.shape[0],data.shape[1],3))

    for i in range((data.shape[0])):
        for j in range((data.shape[1])):
            arr[i,j,0] = color_array[ int(data[i,j]), 0 ]
            arr[i,j,1] = color_array[ int(data[i,j]), 1 ]
            arr[i,j,2] = color_array[ int(data[i,j]), 2 ]
            #print(color_array[ int(data[i,j]) ] )
    return arr

def find_unsegmented_points(masks):
    temp_img1 = np.zeros((rgb_img.shape[0], rgb_img.shape[1]))  # y, x
    for i in range(len(masks)):
        temp_data=masks[i]['segmentation']
        segmentation_data_vals= list(zip(*np.where(temp_data == True)))
        #print(type(segmentation_data_vals))
        for j in segmentation_data_vals:
            temp_img1[j[0],j[1]]=1
    
    #gets the rest of the unsegmented points
    unsegmented_points=  list(zip(*np.where(temp_img1 == 0)))
    

    #visualize which pixels are segmented vs unsegmented
    #unsegmented_points will be set to a single mask for downstream processing
    # fig = plt.figure()
    # plt.imshow(temp_img1)
    # plt.show()
    
    return unsegmented_points

def SpectralAngleMapper(cur_pixel, ref_spec):
        dot_product = np.dot( cur_pixel, ref_spec)
        norm_spectral=np.linalg.norm(cur_pixel)
        norm_ref=np.linalg.norm(ref_spec)
        denom = norm_spectral * norm_ref
        if denom == 0:
            return 3.14
        alpha_rad=math.acos(dot_product / (denom)); 
        return alpha_rad*255/3.1416 
        
def perform_similarity(cube1, spectral_array):
    thread_pool=True # set to true to use thread pool, false to use opencl
    print(cube1.shape, spectral_array.shape)
    #cube 1 x,y,channels
    #spectral array item_num,channels ie 3,164
    if (cube1.shape[2] != spectral_array.shape[1]):
         print("Error: cube and spectral array have different number of channels")
         return
    else :
        kernel_code = """
            __kernel void spectral_angle_mapper(__global const float* spectra,
                                                __global const float* reference,
                                                __global float* angles,
                                                const int num_bands)
            {
                int gid = get_global_id(0);

                float dot_product = 0.0f;
                float norm_spectra = 0.0f;
                float norm_reference = 0.0f;

                for (int i = 0; i < num_bands; i++)
                {
                    dot_product += spectra[gid * num_bands + i] * reference[i];
                    norm_spectra += spectra[gid * num_bands + i] * spectra[gid * num_bands + i];
                    norm_reference += reference[i] * reference[i];
                }

                norm_spectra = sqrt(norm_spectra);
                norm_reference = sqrt(norm_reference);

                angles[gid] = acos(dot_product / (norm_spectra * norm_reference));
            }
            """   
        #perform spectral angle mapper
        spec_sim_arr_1=np.zeros((cube1.shape[0], cube1.shape[1],len(spectral_array)))
        # print(len(spectral_array))
        for k in range(len(spectral_array)):
            print(k)
            ref_spec =  spectral_array[k,:]
            arr=np.zeros((cube1.shape[0], cube1.shape[1])) #y,x
            
            if (thread_pool):
                #thread pool to speed up computation
                with Pool(processes=cpu_count()) as pool:
                        for  i in  range (cube1.shape[0]):
                            itr=0
                            #for result in pool.map(SpectralCorrelationMapper, ( data[i,j,:] for j in range((cube.height)) ) ):
                            for result in pool.starmap(SpectralAngleMapper, ((cube1[i, j, :], ref_spec) for j in range(cube1.shape[1]))):
                                arr[i,itr]=result  
                                itr+=1
                            spec_sim_arr_1[:,:,k]=arr
            else:
                get_test_platforms_and_devices()
                spectra=cube1.astype('float32')
                spectra=spectra.reshape((spectra.shape[0]*spectra.shape[1]),spectra.shape[2])
                # print(spectra.shape, spectra.shape[0])
                reference=ref_spec.astype('float32')
                # Initialize the OpenCL context and command queue
                platform = cl.get_platforms()[0]
                device = platform.get_devices()[0]
                context = cl.Context([device])
                queue = cl.CommandQueue(context)
                # Create OpenCL buffers for the input and output data
                spectra_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=spectra)
                reference_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=reference)
                angles_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=spectra.shape[0] * np.float32().itemsize)

                # Compile the OpenCL kernel program
                program = cl.Program(context, kernel_code).build()

                # Execute the OpenCL kernel function
                program.spectral_angle_mapper(queue, spectra.shape, None, spectra_buffer, reference_buffer, angles_buffer, np.int32(spectra.shape[1]))

                # Read the results from the OpenCL buffer
                angles = np.empty_like(spectra[:, 0], dtype=np.float32)
                cl.enqueue_copy(queue, angles, angles_buffer)
                
                arr=angles.reshape(cube1.shape[0],cube1.shape[1])
                #print(angles.shape, angles.shape[0])
                spec_sim_arr_1[:,:,k]=arr
                        
            # plt.imshow(spec_sim_arr_1[:,:,k]/3.15) 
            # plt.show()
        return spec_sim_arr_1
      
def fuse_results(cube1, masks, unsegmented_points, class_img):
    #use mode to get the class with most values
    temp_img2=np.zeros((cube1.shape[0], cube1.shape[1])) #y,x
    class_img+=1

    for i in range(len(masks)+1):
        
        if (i!=len(masks)):
            temp_data=masks[i]['segmentation']
            segmentation_data_vals= list(zip(*np.where(temp_data == True)))
        else:
            segmentation_data_vals=unsegmented_points
        #print(type(segmentation_data_vals))

        #find mode here 
        temp_array=np.zeros((len(segmentation_data_vals)))
        for j in range(len(segmentation_data_vals)):
            temp_array[j]=class_img[segmentation_data_vals[j][0], segmentation_data_vals[j][1]]
            
        mode_val = int(st.mode(temp_array,keepdims=False).mode)
        #print(mode_val)    
        for j in segmentation_data_vals:
            temp_img2[j[0],j[1]]=mode_val-1
    return temp_img2
    
    
    
    
if __name__ == "__main__":
    
    #print cl enabled devices on machine 
    # for i, platform in enumerate(cl.get_platforms()):
    #     for j, device in enumerate(platform.get_devices()):
    #         print(f"Platform {i}, Device {j}: {device.name}")
            
    
    # clear cuda cache and load sam model
    torch.cuda.empty_cache()
    # model needs to be downloaded from online (https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    # there are other models that can be used as well.
    sam = sam_model_registry["vit_b"](checkpoint="../HyperImages/segment_anything_model/sam_vit_b_01ec64.pth")
    sam.to(device='cuda')
    
    
    # load rgb image into numpy array and convert to rgb
    rgb_img = cv2.imread("images/lena3.png")
    rgb_img = cv2.cvtColor(rgb_img,  cv2.COLOR_BGR2RGB)
    
    #load hyperspectral image into numpy array
    mylist = []
    loaded,mylist = cv2.imreadmulti(mats = mylist, filename = "../HyperImages/img1.tiff", flags = cv2.IMREAD_ANYCOLOR )
    cube1=np.array(mylist)
    cube1 = cube1[:, :, :]  # channels, x, y
    cube1 = np.transpose(cube1, (1, 2, 0)) # x, y, channels 
    # print(cube1.shape)
    
    # create rgb image from hyperspectral image
    rgb_img= extract_rgb(cube1, 163 , 104, 65,  False)

    #resize image to reduce computation time for segmentation
    #need to unresize for correct correlation
    # rgb_img = cv2.resize(rgb_img, (300, 300))
    
    # #perform segmentation
    masks = segment_img(sam,rgb_img)
    
    # # get unsegmented points
    unsegmented_points = find_unsegmented_points(masks)
    
    #load spectral database
    color_array, spectral_array= read_spec_json('json/spectral_database_U20.json')
    
    # create spectral similarity images
    spec_sim_arr_1 = perform_similarity(cube1, spectral_array)

    #convert spectral similarity image to semantic class image
    class_img= semantic_class(spec_sim_arr_1)
    
    #convert semantic class image that has values 0-n to rgb image for visualization 
    color_img= semantic_color(class_img, color_array)
    
    # # fuse the results of spectral angle mapper and segment anything
    temp_img2 = fuse_results(cube1, masks, unsegmented_points, class_img)

    # #convert fused image to rgb image for visualization
    fusion_img= semantic_color(temp_img2, color_array)
    
    #show the resulting image of spectral angle mapper for a layer
    # plt.imshow(spec_sim_arr_1[:,:,1]/3.15)  
    
    #show the resulting hyperspectral classified img
    # plt.imshow(color_img/255)  
    # plt.show()

    #show the resulting image of sam sam 
    plt.imshow(fusion_img/255)  
    plt.show()
    
    #show masks on rgb image
    # plt.imshow(rgb_img)
    # show_anns(masks)
    # plt.show()
