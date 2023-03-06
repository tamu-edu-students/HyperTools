# cmake needs this line
cmake_minimum_required(VERSION 3.10)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_STANDARD 17)
# Define project name
project(user_interface_project C CXX)



find_package(OpenCV  REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${GTK3_INCLUDE_DIRS} )
link_directories(${GTK3_LIBRARY_DIRS} )

#change below to true to use GPU or false to not use GPU
set(use_CUDA true) 

link_libraries(jsoncpp ${GTK3_LIBRARIES} ${OpenCV_LIBS})

if (use_CUDA)


find_package(CUDA)
    if(CUDA_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
        
        add_definitions(${GTK_CFLAGS_OTHER}) 

        add_executable(cuda_example cuda_example.cu)
        add_executable(gtk_cuda_example gtk_cuda_example.cu)
        # below is not necessary right now
        set_target_properties(gtk_cuda_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


    else(CUDA_FOUND)
        message("Required CUDA Packages not found")

    endif(CUDA_FOUND)  

else(use_CUDA)

# Declare the executable target built from your sources
add_executable(image_tool image_tool.cpp)
add_executable(semantic_interface semantic_segmentation.cpp)
add_executable(feature_tool feature_tool.cpp)

add_definitions(${GTK3_CFLAGS} ${GTK3_CFLAGS_OTHER})

endif(use_CUDA)


set (use_cuvis false)

if (use_cuvis)

    include_directories(/lib_cuvis )
    link_directories(/lib/cuvis)
    link_libraries(cuvis)

    add_executable(cubert_example cubert_example.cpp)

    if (use_CUDA)

    add_executable(cuda_cubert_example cuda_cubert_example.cu)

    endif(use_CUDA)

endif()



