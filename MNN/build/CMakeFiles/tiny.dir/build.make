# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lwj-hdu/shz/NAS_KD_NEW/MNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lwj-hdu/shz/NAS_KD_NEW/MNN/build

# Include any dependencies generated for this target.
include CMakeFiles/tiny.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tiny.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tiny.dir/flags.make

CMakeFiles/tiny.dir/main.cpp.o: CMakeFiles/tiny.dir/flags.make
CMakeFiles/tiny.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lwj-hdu/shz/NAS_KD_NEW/MNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tiny.dir/main.cpp.o"
	arm-linux-gnueabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tiny.dir/main.cpp.o -c /home/lwj-hdu/shz/NAS_KD_NEW/MNN/main.cpp

CMakeFiles/tiny.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tiny.dir/main.cpp.i"
	arm-linux-gnueabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lwj-hdu/shz/NAS_KD_NEW/MNN/main.cpp > CMakeFiles/tiny.dir/main.cpp.i

CMakeFiles/tiny.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tiny.dir/main.cpp.s"
	arm-linux-gnueabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lwj-hdu/shz/NAS_KD_NEW/MNN/main.cpp -o CMakeFiles/tiny.dir/main.cpp.s

# Object files for target tiny
tiny_OBJECTS = \
"CMakeFiles/tiny.dir/main.cpp.o"

# External object files for target tiny
tiny_EXTERNAL_OBJECTS =

tiny: CMakeFiles/tiny.dir/main.cpp.o
tiny: CMakeFiles/tiny.dir/build.make
tiny: /usr/local/lib/libopencv_dnn.so.3.4.15
tiny: /usr/local/lib/libopencv_highgui.so.3.4.15
tiny: /usr/local/lib/libopencv_ml.so.3.4.15
tiny: /usr/local/lib/libopencv_objdetect.so.3.4.15
tiny: /usr/local/lib/libopencv_shape.so.3.4.15
tiny: /usr/local/lib/libopencv_stitching.so.3.4.15
tiny: /usr/local/lib/libopencv_superres.so.3.4.15
tiny: /usr/local/lib/libopencv_videostab.so.3.4.15
tiny: /usr/local/lib/libopencv_calib3d.so.3.4.15
tiny: /usr/local/lib/libopencv_features2d.so.3.4.15
tiny: /usr/local/lib/libopencv_flann.so.3.4.15
tiny: /usr/local/lib/libopencv_photo.so.3.4.15
tiny: /usr/local/lib/libopencv_video.so.3.4.15
tiny: /usr/local/lib/libopencv_videoio.so.3.4.15
tiny: /usr/local/lib/libopencv_imgcodecs.so.3.4.15
tiny: /usr/local/lib/libopencv_imgproc.so.3.4.15
tiny: /usr/local/lib/libopencv_core.so.3.4.15
tiny: CMakeFiles/tiny.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lwj-hdu/shz/NAS_KD_NEW/MNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tiny"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tiny.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tiny.dir/build: tiny

.PHONY : CMakeFiles/tiny.dir/build

CMakeFiles/tiny.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tiny.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tiny.dir/clean

CMakeFiles/tiny.dir/depend:
	cd /home/lwj-hdu/shz/NAS_KD_NEW/MNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lwj-hdu/shz/NAS_KD_NEW/MNN /home/lwj-hdu/shz/NAS_KD_NEW/MNN /home/lwj-hdu/shz/NAS_KD_NEW/MNN/build /home/lwj-hdu/shz/NAS_KD_NEW/MNN/build /home/lwj-hdu/shz/NAS_KD_NEW/MNN/build/CMakeFiles/tiny.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tiny.dir/depend

