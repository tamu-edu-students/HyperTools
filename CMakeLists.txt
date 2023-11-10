# cmake needs this line
cmake_minimum_required(VERSION 3.16)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_STANDARD 17)
# Define project name
project(user_interface_project C CXX)

#change below to true to use GPU or false to not use GPU
set(use_CUDA false) 
#change below to true to use cuvis.sdk
set (use_cuvis true)


find_package(OpenCV  REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${GTK3_INCLUDE_DIRS} )
link_directories(${GTK3_LIBRARY_DIRS} )
link_libraries(jsoncpp ${GTK3_LIBRARIES} ${OpenCV_LIBS})

if (use_CUDA)

    find_package(CUDA)
        if(CUDA_FOUND)
            enable_language(CUDA)
            #set(CMAKE_CUDA_STANDARD 17)

            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES 75)
            endif()
            
            add_definitions(${GTK_CFLAGS_OTHER}) 
            cmake_policy(SET CMP0146 NEW) 

            add_executable(cuda_example examples/cuda_example.cu)
            set_target_properties(cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            add_executable(gtk_cuda_example examples/gtk_cuda_example.cu)
            set_target_properties(gtk_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


        else(CUDA_FOUND)
            message("Required CUDA Packages not found")

        endif(CUDA_FOUND)  

else(use_CUDA)

    # Declare the executable target built from your sources
    # add_executable(semantic_interface examples/semantic_segmentation.cpp)
    add_executable(feature_tool examples/feature_tool.cpp)
    add_executable(feature_testing examples/feature_testing.cpp)
    # add_executable(ground_truth_example examples/ground_truth_example.cpp)

    add_definitions(${GTK3_CFLAGS} ${GTK3_CFLAGS_OTHER})

endif(use_CUDA)


if (use_cuvis)

    # list(APPEND CMAKE_MODULE_PATH "/cuvis.sdk/cuvis.cpp/")
    # list(APPEND CMAKE_MODULE_PATH "/cuvis.sdk/cuvis.c/")
    set(DOXYGEN_BUILD_DOCUMENTATION OFF CACHE BOOL "Create and install the HTML based API documentation (requires Doxygen)" FORCE)
    list(APPEND CMAKE_MODULE_PATH "/workspaces/HyperTools/submodules/cuvis.sdk/cuvis.c/")
    list(APPEND CMAKE_MODULE_PATH "/workspaces/HyperTools/submodules/cuvis.sdk/cuvis.cpp/")
    add_definitions(-Duse_cuvis)
    
    
    find_package(Cuvis REQUIRED 3.2.0)
    find_package(CuvisCpp REQUIRED)    
    

    add_executable(cubert_example examples/cubert_example.cpp)
    target_link_libraries(cubert_example PRIVATE cuvis::c)

    # # add_executable(deeplab_example examples/deeplab_example.cpp)
    # # target_link_libraries(deeplab_example PRIVATE cuvis::cpp)

    # add_executable(image_tool_cuvis examples/image_tool.cpp)
    # target_link_libraries(image_tool_cuvis PRIVATE cuvis::cpp)


    if (use_CUDA)

        add_executable(cuda_cubert_example examples/cuda_cubert_example.cu)
        target_link_libraries(cuda_cubert_example PRIVATE cuvis::cpp)

    endif(use_CUDA)
else (use_cuvis)

    add_executable(image_tool examples/image_tool.cpp)

endif()


