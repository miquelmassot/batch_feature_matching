cmake_minimum_required(VERSION 2.8)
project(batch_feature_matching)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system REQUIRED)
include_directories(${Opencv_INCLUDE_DIRS}include)
add_executable(batch_matcher src/batch_main.cpp src/batch_feature_matcher.cpp)
target_link_libraries(batch_matcher
                      ${OpenCV_LIBS}
                      ${Boost_FILESYSTEM_LIBRARY}
                      ${Boost_SYSTEM_LIBRARY})