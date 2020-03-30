cmake_minimum_required (VERSION 2.6)

set (PROJ_NAME retinaface_ncnn)
project(${PROJ_NAME})
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(zlib)

find_package( OpenCV REQUIRED )

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")

#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-msse4")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-msse4")

include_directories(ncnn/include)
include_directories(${CMAKE_SOURCE_DIR}/zlib)
include_directories(${CMAKE_SOURCE_DIR})
add_executable(${PROJ_NAME}
			   src/anchor_generator.cpp
			   src/config.cpp
			   src/detect.cpp
			   src/tools.cpp
			   src/RR_Lib.cpp 
			   src/RR_Lib.h
			   src/anchor_generator.h
			   src/config.h
			   cnpy.cpp
			   )
target_link_libraries(${PROJ_NAME}
						${OpenCV_LIBS}
						zlib
                        optimized  ${PROJECT_SOURCE_DIR}/ncnn/lib/Release/ncnn.lib
                        debug ${PROJECT_SOURCE_DIR}/ncnn/lib/Debug/ncnn.lib
                        )

