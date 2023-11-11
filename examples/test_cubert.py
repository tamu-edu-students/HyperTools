import sys
sys.path.append("/workspaces/HyperTools/submodules/cuvis.sdk/cuvis.pyil/build")
import cuvis_il
import cuvis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def loadMeasurement(userSettingsDir, measurementLoc, visualize=False):

    print("loading user settings...")
    settings = cuvis.General(userSettingsDir)
    settings.setLogLevel("info")

    print("loading measurement file...")
    mesu = cuvis.Measurement(measurementLoc)
    print("Data 1 {} t={}ms mode={}".format(mesu.Name,mesu.IntegrationTime,mesu.ProcessingMode,))

    if not isinstance(mesu.MeasurementFlags, list):
        mesu.MeasurementFlags = [mesu.MeasurementFlags]

    if len(mesu.MeasurementFlags) > 0:
        print("Flags")
        for flag in mesu.MeasurementFlags:
            print(" - {} ({})".format(flag, flag)) 

    assert mesu.ProcessingMode == "Raw", \
        "This example requires Raw mode!"

    cube = mesu.Data.pop("cube", None)
    if cube is None:
        raise Exception("Cube not found")
   
    if visualize==True:
        # below is to visualize the image
        #print((cube.array.shape))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(cube.array[:,:, 0],cmap=plt.cm.gray, animated=True)

        def animated_image(i):
            # cube.array[x, y, chn]
            #for chn in np.arange(cube.channels):
            data = cube.array[:,:, i]
            x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
            im.set_array(x_norm*50)

            plt.title("Hyperspectral Image Layer {} of {} with wavelentgth {}".format(i, mesu.Name, cube.wavelength[i]))

        print(type(cube.array[5,5, 5]))    
        ani = animation.FuncAnimation(fig, animated_image,frames=cube.channels, interval=20, repeat=False)
        plt.show()

    #print("finished.")
    return cube

def   reprocessMeasurement(userSettingsDir,measurementLoc,darkLoc,whiteLoc,distanceLoc,factoryDir,outDir, visualize=False):    
    
    print("loading user settings...")
    settings = cuvis.General(userSettingsDir)
    settings.setLogLevel("info")

    print("loading measurement file...")
    mesu = cuvis.Measurement(measurementLoc)

    print("loading dark...")
    dark = cuvis.Measurement(darkLoc)
    print("loading white...")
    white = cuvis.Measurement(whiteLoc)
    print("loading dark...")
    distance = cuvis.Measurement(distanceLoc)

    print("Data 1 {} t={}ms mode={}".format(mesu.Name,mesu.IntegrationTime,mesu.ProcessingMode,))

    print("loading calibration and processing context (factory)...")
    calibration = cuvis.Calibration(factoryDir)
    processingContext = cuvis.ProcessingContext(calibration)

    print("set references...")
    processingContext.setReference(dark, "Dark")
    processingContext.setReference(white, "White")
    processingContext.setReference(distance, "Distance")

    modes = [
             "Reflectance"
             ]

    procArgs = cuvis.CubertProcessingArgs()
    saveArgs = cuvis.CubertSaveArgs(AllowOverwrite=True)

    for mode in modes:

        procArgs.ProcessingMode = mode
        isCapable = processingContext.isCapable(mesu, procArgs)

        if isCapable:
            print("processing to mode {}...".format(mode))
            processingContext.setProcessingArgs(procArgs)
            mesu = processingContext.apply(mesu)
            cube = mesu.Data.pop("cube", None)
            
            if cube is None:
                raise Exception("Cube not found")

        else:
            print("Cannot process to {} mode!".format(mode))
            
            
    if visualize==True:
        # below is to visualize the image
        #print((cube.array.shape))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(cube.array[:,:, 0],cmap=plt.cm.gray, animated=True)

        def animated_image(i):
            # cube.array[x, y, chn]
            #for chn in np.arange(cube.channels):
            data = cube.array[:,:, i]
            #x_norm = data*10
            x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
            #im.set_array(x_norm*65535)
            #converts to 8 bit 
            img8=(x_norm*255).astype('uint8')
            im.set_array(img8*256)

            plt.title("Hyperspectral Image Layer {} of {} with wavelentgth {}".format(i, mesu.Name, cube.wavelength[i]))

        ani = animation.FuncAnimation(fig, animated_image,frames=cube.channels, interval=20, repeat=False)
        plt.show()
    
    #print("finished.")
    return cube

def extract_rgb(cube, red_layer=78 , green_layer=40, blue_layer=25,  visualize=False):

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
  
  
  
if __name__ == "__main__":


   # make sure terminal is in the right directory when running this file
   # I was running from the examples directory of hypertools
    
    userSettingsDir = "../settings/" 
    measurementLoc = "../../HyperImages/cornfields/session_002/session_002_490.cu3"
    darkLoc = "../../HyperImages/cornfields/Calibration/dark__session_002_003_snapshot16423119279414228.cu3"
    whiteLoc = "../../HyperImages/cornfields/Calibration/white__session_002_752_snapshot16423136896447489.cu3"
    distanceLoc = "../../HyperImages/cornfields/Calibration/distanceCalib__session_000_790_snapshot16423004058237746.cu3"
    factoryDir = "../settings/" # init.daq file
    outDir ="../../HyperImages/export/"
    
    # use below if image is already processed
    # cube= loadMeasurement(userSettingsDir, measurementLoc, False)
    # use below is raw mode is required
    cube = reprocessMeasurement(userSettingsDir,measurementLoc,darkLoc,whiteLoc,distanceLoc,factoryDir,outDir, False)
    data = cube.array[:,:, :] # x,y,chan
    # print(data.shape)
    rgb_img= extract_rgb(cube,visualize=True)
