import numpy as np
import cuvis
import cv2 as cv
import matplotlib.pyplot as plt
import glob

def extract_rgb(cube, red_layer=78 , green_layer=40, blue_layer=25,  visualize=False):

    
    red_img = cube[ red_layer,:,:]
    green_img = cube[ green_layer,:,:]
    blue_img = cube[ blue_layer,:,:]

        
    data=np.stack([red_img,green_img,blue_img], axis=-1)
    # print(data.shape)
    #print(type(image))

    #convert to 8bit
    x_norm = (data-np.min(data))/(np.max(data)-np.min(data))
    image=(x_norm*255).astype('uint8')
    if visualize:
        #pass
        plt.imshow(image)
        plt.show()
    return image  

def   reprocessMeasurement_cu3s(userSettingsDir,measurementLoc,darkLoc,whiteLoc,distance,factoryDir):    
    
    settings = cuvis.General(userSettingsDir)
    # settings.set_log_level("info")

    sessionM = cuvis.SessionFile(measurementLoc)
    mesu = sessionM[0]
    assert mesu._handle
    
    sessionDk = cuvis.SessionFile(darkLoc)
    dark = sessionDk[0]
    assert dark._handle
    
    sessionWt = cuvis.SessionFile(whiteLoc)
    white = sessionWt[0]
    assert white._handle
    
    # sessionDc = cuvis.SessionFile(distanceLoc)
    # distance = sessionDc[0]
    # assert distance._handle
    
    processingContext = cuvis.ProcessingContext(sessionM)
    
    processingContext.calc_distance(distance)
    processingContext.processing_mode = cuvis.ProcessingMode.Reflectance
    
    processingContext.set_reference(dark, cuvis.ReferenceType.Dark)
    processingContext.set_reference(white, cuvis.ReferenceType.White)
    
    assert processingContext.is_capable(mesu,
                                       processingContext.get_processing_args())
    
    processingContext.apply(mesu)
    cube = mesu.data.get("cube", None)
    
    #print("finished.")
    cube_result = cube.array
    cube_result = np.transpose(cube_result, (2, 0, 1))
    # print(cube_result.shape)
    return cube_result



userSettingsDir = "settings/ultris5/" 
measurementLoc = "../HyperImages/Checkerboard_001.cu3s"
darkLoc = "../HyperImages/Dark_001.cu3s"
whiteLoc = "../HyperImages/White_001.cu3s"
factoryDir = "settings/ultris5/"  # init.daq file
distance = 500 # dist in mm      # this was a guess


cube1= reprocessMeasurement_cu3s(userSettingsDir,measurementLoc,darkLoc,whiteLoc,distance,factoryDir)

# changing values changes the extracted layers
image0=extract_rgb(cube1,40,20,2)
# image0=extract_rgb(cube1, 50, 30, 10)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# images = glob.glob('Checkerboard_001.cu3s')
images = image0
for fname in images:
    img = image0
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)
# img = cv.imread('Checkerboard_001.cu3s')
img = image0
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# camera calibration parameter
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

# Extracting focal length from the camera matrix
focal_length_x = mtx[0, 0]  # Focal length along the x-axis
focal_length_y = mtx[1, 1]  # Focal length along the y-axis

print("Focal Length (fx):", focal_length_x)
print("Focal Length (fy):", focal_length_y)

# Extracting optical center parameters from the camera matrix
optical_center_x = mtx[0, 2]  # Principal point x-coordinate
optical_center_y = mtx[1, 2]  # Principal point y-coordinate

print("Optical Center (cx):", optical_center_x)
print("Optical Center (cy):", optical_center_y)