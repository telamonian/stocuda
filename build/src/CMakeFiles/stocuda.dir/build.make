# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

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
CMAKE_COMMAND = "/Applications/CMake 2.8-10.app/Contents/bin/cmake"

# The command to remove a file.
RM = "/Applications/CMake 2.8-10.app/Contents/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = "/Applications/CMake 2.8-10.app/Contents/bin/ccmake"

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/tel/git/stocuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/tel/git/stocuda/build

# Include any dependencies generated for this target.
include src/CMakeFiles/stocuda.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/stocuda.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/stocuda.dir/flags.make

src/CMakeFiles/stocuda.dir/hazard.cc.o: src/CMakeFiles/stocuda.dir/flags.make
src/CMakeFiles/stocuda.dir/hazard.cc.o: ../src/hazard.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/tel/git/stocuda/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/stocuda.dir/hazard.cc.o"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/stocuda.dir/hazard.cc.o -c /Users/tel/git/stocuda/src/hazard.cc

src/CMakeFiles/stocuda.dir/hazard.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stocuda.dir/hazard.cc.i"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/tel/git/stocuda/src/hazard.cc > CMakeFiles/stocuda.dir/hazard.cc.i

src/CMakeFiles/stocuda.dir/hazard.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stocuda.dir/hazard.cc.s"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/tel/git/stocuda/src/hazard.cc -o CMakeFiles/stocuda.dir/hazard.cc.s

src/CMakeFiles/stocuda.dir/hazard.cc.o.requires:
.PHONY : src/CMakeFiles/stocuda.dir/hazard.cc.o.requires

src/CMakeFiles/stocuda.dir/hazard.cc.o.provides: src/CMakeFiles/stocuda.dir/hazard.cc.o.requires
	$(MAKE) -f src/CMakeFiles/stocuda.dir/build.make src/CMakeFiles/stocuda.dir/hazard.cc.o.provides.build
.PHONY : src/CMakeFiles/stocuda.dir/hazard.cc.o.provides

src/CMakeFiles/stocuda.dir/hazard.cc.o.provides.build: src/CMakeFiles/stocuda.dir/hazard.cc.o

src/CMakeFiles/stocuda.dir/pnet.cc.o: src/CMakeFiles/stocuda.dir/flags.make
src/CMakeFiles/stocuda.dir/pnet.cc.o: ../src/pnet.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/tel/git/stocuda/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/stocuda.dir/pnet.cc.o"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/stocuda.dir/pnet.cc.o -c /Users/tel/git/stocuda/src/pnet.cc

src/CMakeFiles/stocuda.dir/pnet.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stocuda.dir/pnet.cc.i"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/tel/git/stocuda/src/pnet.cc > CMakeFiles/stocuda.dir/pnet.cc.i

src/CMakeFiles/stocuda.dir/pnet.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stocuda.dir/pnet.cc.s"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/tel/git/stocuda/src/pnet.cc -o CMakeFiles/stocuda.dir/pnet.cc.s

src/CMakeFiles/stocuda.dir/pnet.cc.o.requires:
.PHONY : src/CMakeFiles/stocuda.dir/pnet.cc.o.requires

src/CMakeFiles/stocuda.dir/pnet.cc.o.provides: src/CMakeFiles/stocuda.dir/pnet.cc.o.requires
	$(MAKE) -f src/CMakeFiles/stocuda.dir/build.make src/CMakeFiles/stocuda.dir/pnet.cc.o.provides.build
.PHONY : src/CMakeFiles/stocuda.dir/pnet.cc.o.provides

src/CMakeFiles/stocuda.dir/pnet.cc.o.provides.build: src/CMakeFiles/stocuda.dir/pnet.cc.o

src/CMakeFiles/stocuda.dir/stocuda.cc.o: src/CMakeFiles/stocuda.dir/flags.make
src/CMakeFiles/stocuda.dir/stocuda.cc.o: ../src/stocuda.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/tel/git/stocuda/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/stocuda.dir/stocuda.cc.o"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/stocuda.dir/stocuda.cc.o -c /Users/tel/git/stocuda/src/stocuda.cc

src/CMakeFiles/stocuda.dir/stocuda.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stocuda.dir/stocuda.cc.i"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/tel/git/stocuda/src/stocuda.cc > CMakeFiles/stocuda.dir/stocuda.cc.i

src/CMakeFiles/stocuda.dir/stocuda.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stocuda.dir/stocuda.cc.s"
	cd /Users/tel/git/stocuda/build/src && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/tel/git/stocuda/src/stocuda.cc -o CMakeFiles/stocuda.dir/stocuda.cc.s

src/CMakeFiles/stocuda.dir/stocuda.cc.o.requires:
.PHONY : src/CMakeFiles/stocuda.dir/stocuda.cc.o.requires

src/CMakeFiles/stocuda.dir/stocuda.cc.o.provides: src/CMakeFiles/stocuda.dir/stocuda.cc.o.requires
	$(MAKE) -f src/CMakeFiles/stocuda.dir/build.make src/CMakeFiles/stocuda.dir/stocuda.cc.o.provides.build
.PHONY : src/CMakeFiles/stocuda.dir/stocuda.cc.o.provides

src/CMakeFiles/stocuda.dir/stocuda.cc.o.provides.build: src/CMakeFiles/stocuda.dir/stocuda.cc.o

# Object files for target stocuda
stocuda_OBJECTS = \
"CMakeFiles/stocuda.dir/hazard.cc.o" \
"CMakeFiles/stocuda.dir/pnet.cc.o" \
"CMakeFiles/stocuda.dir/stocuda.cc.o"

# External object files for target stocuda
stocuda_EXTERNAL_OBJECTS =

src/stocuda.so: src/CMakeFiles/stocuda.dir/hazard.cc.o
src/stocuda.so: src/CMakeFiles/stocuda.dir/pnet.cc.o
src/stocuda.so: src/CMakeFiles/stocuda.dir/stocuda.cc.o
src/stocuda.so: src/CMakeFiles/stocuda.dir/build.make
src/stocuda.so: /usr/local/lib/libboost_python-mt.dylib
src/stocuda.so: /usr/local/lib/libpython2.7.dylib
src/stocuda.so: src/libhazard.a
src/stocuda.so: /Developer/NVIDIA/CUDA-5.0/lib/libcudart.dylib
src/stocuda.so: src/CMakeFiles/stocuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared module stocuda.so"
	cd /Users/tel/git/stocuda/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stocuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/stocuda.dir/build: src/stocuda.so
.PHONY : src/CMakeFiles/stocuda.dir/build

src/CMakeFiles/stocuda.dir/requires: src/CMakeFiles/stocuda.dir/hazard.cc.o.requires
src/CMakeFiles/stocuda.dir/requires: src/CMakeFiles/stocuda.dir/pnet.cc.o.requires
src/CMakeFiles/stocuda.dir/requires: src/CMakeFiles/stocuda.dir/stocuda.cc.o.requires
.PHONY : src/CMakeFiles/stocuda.dir/requires

src/CMakeFiles/stocuda.dir/clean:
	cd /Users/tel/git/stocuda/build/src && $(CMAKE_COMMAND) -P CMakeFiles/stocuda.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/stocuda.dir/clean

src/CMakeFiles/stocuda.dir/depend:
	cd /Users/tel/git/stocuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/tel/git/stocuda /Users/tel/git/stocuda/src /Users/tel/git/stocuda/build /Users/tel/git/stocuda/build/src /Users/tel/git/stocuda/build/src/CMakeFiles/stocuda.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/stocuda.dir/depend

