cmake_minimum_required(VERSION 3.10)
project(MNN) #项目名称

set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_MODULE_PATH /home/lwj-hdu/opencv/build/cmake)

#告知当前使用的是交叉编译方式，必须配置
SET(CMAKE_SYSTEM_NAME Linux)
#指定C交叉编译器,必须配置 
set(CMAKE_C_COMPILER   arm-linux-gnueabi-gcc)
#指定C++交叉编译器
set(CMAKE_CXX_COMPILER arm-linux-gnueabi-g++)

#CMAKE_SYSROOT只有在3.0以上的版本才有效
# SET(CMAKE_SYSROOT "/opt/arm-oe-linux-gnueabi/armv7")
#SET(MYSYSROOT "/opt/arm-oe-linux-gnueabi/armv7")
#指定交叉编译环境安装目录...
SET(CMAKE_FIND_ROOT_PATH "/home/lwj-hdu/buildtools/armv7/gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabi/bin")
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_VERBOSE_MAKEFILEON ON)

SET(CMAKE_SYSROOT "/home/lwj-hdu/opencv-arm/build/bin")
SET(CMAKE_SYSROOT "/home/lwj-hdu/opencv-arm/build/include")
find_package(OpenCV REQUIRED)

if(Opencv_FOUND)
    message(STATUS "The Opecv lib is found!") 
endif()



set(MNN_DIR /home/lwj-hdu/MNN)
include_directories(${MNN_DIR}/include)
include_directories(${MNN_DIR}/include/MNN)
include_directories(${MNN_DIR}/tools)
include_directories(${MNN_DIR}/tools/cpp)
include_directories(${MNN_DIR}/source)
include_directories(${MNN_DIR}/source/backend)
include_directories(${MNN_DIR}/source/core)

LINK_DIRECTORIES(${MNN_DIR}/build)

#把当前目录(.)下所有源代码文件和头文件加入变量SRC_LIST
# AUX_SOURCE_DIRECTORY(. SRC_LIST)
# #生成应用程序 tiny
# ADD_EXECUTABLE(tiny ${SRC_LIST})
add_executable(niu2 main.cpp)
# target_link_libraries(tiny  ${PROJECT_SOURCE_DIR}/)
target_link_libraries(niu2 -lMNN ${OpenCV_LIBS})