# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /new_disk/liuxh/RIFT++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /new_disk/liuxh/RIFT++

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /new_disk/liuxh/RIFT++/CMakeFiles /new_disk/liuxh/RIFT++/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /new_disk/liuxh/RIFT++/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named RIFT

# Build rule for target.
RIFT: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 RIFT
.PHONY : RIFT

# fast build rule for target.
RIFT/fast:
	$(MAKE) -f CMakeFiles/RIFT.dir/build.make CMakeFiles/RIFT.dir/build
.PHONY : RIFT/fast

PhaseCongruency/phase.o: PhaseCongruency/phase.cpp.o

.PHONY : PhaseCongruency/phase.o

# target to build an object file
PhaseCongruency/phase.cpp.o:
	$(MAKE) -f CMakeFiles/RIFT.dir/build.make CMakeFiles/RIFT.dir/PhaseCongruency/phase.cpp.o
.PHONY : PhaseCongruency/phase.cpp.o

PhaseCongruency/phase.i: PhaseCongruency/phase.cpp.i

.PHONY : PhaseCongruency/phase.i

# target to preprocess a source file
PhaseCongruency/phase.cpp.i:
	$(MAKE) -f CMakeFiles/RIFT.dir/build.make CMakeFiles/RIFT.dir/PhaseCongruency/phase.cpp.i
.PHONY : PhaseCongruency/phase.cpp.i

PhaseCongruency/phase.s: PhaseCongruency/phase.cpp.s

.PHONY : PhaseCongruency/phase.s

# target to generate assembly for a file
PhaseCongruency/phase.cpp.s:
	$(MAKE) -f CMakeFiles/RIFT.dir/build.make CMakeFiles/RIFT.dir/PhaseCongruency/phase.cpp.s
.PHONY : PhaseCongruency/phase.cpp.s

TestRift/TestPhase.o: TestRift/TestPhase.cpp.o

.PHONY : TestRift/TestPhase.o

# target to build an object file
TestRift/TestPhase.cpp.o:
	$(MAKE) -f CMakeFiles/RIFT.dir/build.make CMakeFiles/RIFT.dir/TestRift/TestPhase.cpp.o
.PHONY : TestRift/TestPhase.cpp.o

TestRift/TestPhase.i: TestRift/TestPhase.cpp.i

.PHONY : TestRift/TestPhase.i

# target to preprocess a source file
TestRift/TestPhase.cpp.i:
	$(MAKE) -f CMakeFiles/RIFT.dir/build.make CMakeFiles/RIFT.dir/TestRift/TestPhase.cpp.i
.PHONY : TestRift/TestPhase.cpp.i

TestRift/TestPhase.s: TestRift/TestPhase.cpp.s

.PHONY : TestRift/TestPhase.s

# target to generate assembly for a file
TestRift/TestPhase.cpp.s:
	$(MAKE) -f CMakeFiles/RIFT.dir/build.make CMakeFiles/RIFT.dir/TestRift/TestPhase.cpp.s
.PHONY : TestRift/TestPhase.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... RIFT"
	@echo "... PhaseCongruency/phase.o"
	@echo "... PhaseCongruency/phase.i"
	@echo "... PhaseCongruency/phase.s"
	@echo "... TestRift/TestPhase.o"
	@echo "... TestRift/TestPhase.i"
	@echo "... TestRift/TestPhase.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

