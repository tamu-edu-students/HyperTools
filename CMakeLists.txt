# cmake needs this line
cmake_minimum_required(VERSION 3.10)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# Define project name
project(user_interface_project C CXX)



find_package(OpenCV  REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)



include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${GTK3_INCLUDE_DIRS} )
link_directories(${GTK3_LIBRARY_DIRS} )

# Declare the executable target built from your sources
add_executable(image_tool image_tool.cpp)
add_executable(semantic_interface semantic_segmentation.cpp)
add_executable(feature_tool feature_tool.cpp)
add_executable(test test.cpp)

# Link your application with OpenCV libraries
target_link_libraries(image_tool PRIVATE ${OpenCV_LIBS})
target_link_libraries(image_tool PRIVATE ${GTK3_LIBRARIES} )
target_link_libraries(image_tool PRIVATE jsoncpp)

target_link_libraries(semantic_interface PRIVATE ${OpenCV_LIBS})
target_link_libraries(semantic_interface PRIVATE ${GTK3_LIBRARIES} )
target_link_libraries(semantic_interface PRIVATE jsoncpp)

target_link_libraries(feature_tool PRIVATE ${OpenCV_LIBS})
target_link_libraries(feature_tool PRIVATE ${GTK3_LIBRARIES} )
target_link_libraries(feature_tool PRIVATE jsoncpp)

target_link_libraries(test PRIVATE ${OpenCV_LIBS})
target_link_libraries(test PRIVATE ${GTK3_LIBRARIES} )
target_link_libraries(test PRIVATE jsoncpp)

add_definitions(${GTK3_CFLAGS} ${GTK3_CFLAGS_OTHER})







