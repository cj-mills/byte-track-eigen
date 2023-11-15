cmake_minimum_required(VERSION 3.20)
project(byte-track-eigen VERSION 1.0.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add library
add_library(ByteTrackEigen SHARED
    src/BaseTrack.cpp
    src/BoundingBoxTrackUtils.cpp
    src/BYTETracker.cpp
    src/HungarianAlgorithmEigen.cpp
    src/KalmanBBoxTrack.cpp
    src/KalmanFilter.cpp
    src/LinearAssignment.cpp
    src/BoundingBoxIoUMatching.cpp
)

# Compiler options for different build types
target_compile_options(ByteTrackEigen PRIVATE 
  $<$<CONFIG:Debug>:-DDEBUG -g>
  $<$<CONFIG:Release>:-O3>
)

# Set different runtime output directories for Debug and Release
set_target_properties(ByteTrackEigen PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
)

# Eigen include directory
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eigen")

# Create the external directory if it does not exist
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/")
  file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/external/")
endif()

# Check if Eigen directory exists
if(NOT EXISTS ${EIGEN3_INCLUDE_DIR})
    # Eigen URL and expected ZIP file path
    set(EIGEN3_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip")
    set(EIGEN3_ZIP "${CMAKE_BINARY_DIR}/eigen-3.4.0.zip")

    # Download Eigen
    message(STATUS "Downloading Eigen to ${EIGEN3_ZIP}")
    file(DOWNLOAD ${EIGEN3_URL} ${EIGEN3_ZIP}
         STATUS EIGEN3_DOWNLOAD_STATUS
         TIMEOUT 60) # Timeout in seconds

    # Check download status
    list(GET EIGEN3_DOWNLOAD_STATUS 0 EIGEN3_DOWNLOAD_RESULT)
    if(EIGEN3_DOWNLOAD_RESULT EQUAL 0)
        # Extract Eigen
        message(STATUS "Extracting Eigen")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${EIGEN3_ZIP}
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                        RESULT_VARIABLE EIGEN3_EXTRACT_RESULT)

        # Check extraction status and rename directory
        if(EIGEN3_EXTRACT_RESULT EQUAL 0)
            file(RENAME "${CMAKE_BINARY_DIR}/eigen-3.4.0" ${EIGEN3_INCLUDE_DIR})
        else()
            message(FATAL_ERROR "Failed to extract Eigen")
        endif()
    else()
        message(FATAL_ERROR "Failed to download Eigen")
    endif()
endif()

# Include Eigen and project headers
target_include_directories(ByteTrackEigen PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(ByteTrackEigen PRIVATE include)

# Definitions for DLL export, if needed
target_compile_definitions(ByteTrackEigen PRIVATE BUILDING_BYTE_TRACK_EIGEN)
