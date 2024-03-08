# cmake needs this line
cmake_minimum_required(VERSION 3.20)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_STANDARD 17)

#change below to true to use GPU or false to not use GPU
set(use_CUDA true) 
#change below to true to use cuvis.sdk
set (use_cuvis true)

if (use_CUDA)
# Define project name
project(hypertools_project C CXX CUDA)
else(use_CUDA)
project(hypertools_project C CXX )
endif(use_CUDA)

find_package(OpenCV  REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GDAL REQUIRED)
set(MATIO_LIBRARIES "/usr/local/lib/libmatio.so")
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${GTK3_INCLUDE_DIRS} )
include_directories(${MATIO_INCLUDE_DIRS})
link_directories(${GTK3_LIBRARY_DIRS} )
link_libraries(jsoncpp ${GTK3_LIBRARIES} ${OpenCV_LIBS} ${GDAL_LIBRARY} ${MATIO_LIBRARIES})

if (use_cuvis)

    list(APPEND CMAKE_MODULE_PATH "/cuvis.sdk/cuvis.cpp/")
    list(APPEND CMAKE_MODULE_PATH "/cuvis.sdk/cuvis.c/")
    set(DOXYGEN_BUILD_DOCUMENTATION OFF CACHE BOOL "Create and install the HTML based API documentation (requires Doxygen)" FORCE)

    add_definitions(-Duse_cuvis)
    find_package(Cuvis REQUIRED 3.2.0)
    find_package(CuvisCpp REQUIRED)   
    else (use_cuvis)
    
endif()

if (use_CUDA)
    add_definitions(-Duse_cuda)
    enable_language(CUDA)
    set (CMAKE_CUDA_STANDARD 17)
    set (CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc) 
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86) # should be modified to match gpu architecture, 86 is for RTX 30xx series
    endif()
            
    add_definitions(${GTK_CFLAGS_OTHER}) 

    add_executable(cuda_example examples/cuda_example.cu)
    set_target_properties(cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

   

    # add_executable(gtk_cuda_example examples/gtk_cuda_example.cu)
    # set_target_properties(gtk_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if (use_cuvis)
        set_source_files_properties(examples/cuda_cubert_example.cu PROPERTIES LANGUAGE CUDA)
        add_executable(cuda_cubert_example examples/cuda_cubert_example.cu)
        set_target_properties(cuda_cubert_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(cuda_cubert_example PRIVATE cuvis::cpp)


        set_source_files_properties(examples/real_time_class.cu PROPERTIES LANGUAGE CUDA)
        add_executable(cuda_real_time_class examples/real_time_class.cu)
        set_target_properties(cuda_real_time_class PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(cuda_real_time_class PRIVATE cuvis::cpp)


    else (use_cuvis)
        set_source_files_properties(examples/ground_truth_example_envi.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(gt_stats_cuda examples/ground_truth_example_envi.cpp)
        set_target_properties(gt_stats_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(gt_stats_cuda stdc++fs)
    endif()

else(use_CUDA)

    # Declare the executable target built from your sources
    add_definitions(${GTK3_CFLAGS} ${GTK3_CFLAGS_OTHER})

    if (use_cuvis)
        add_executable(image_tool_cuvis examples/image_tool.cpp)
        target_link_libraries(image_tool_cuvis PRIVATE cuvis::cpp)

        add_executable(cubert_example examples/cubert_example.cpp)
        target_link_libraries(cubert_example PRIVATE cuvis::cpp)

        # # add_executable(deeplab_example examples/deeplab_example.cpp)
        # # target_link_libraries(deeplab_example PRIVATE cuvis::cpp)

    else (use_cuvis)
        # add_executable(semantic_interface examples/semantic_segmentation.cpp)
        # add_executable(feature_tool examples/feature_tool.cpp)
        add_executable(feature_testing examples/feature_testing.cpp)
        add_executable(ground_truth_example_envi examples/ground_truth_example_envi.cpp)
        # add_executable(ground_truth_example examples/ground_truth_example.cpp)
        add_executable(image_tool examples/image_tool.cpp)

    endif()

endif(use_CUDA)
