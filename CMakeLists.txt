##
#  CMake script for the step-8 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "main")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  main.cpp
  solver.cpp
  )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.1.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fulfilled?
#
IF(NOT DEAL_II_TRILINOS_WITH_SACADO OR NOT DEAL_II_WITH_MUPARSER) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_MUPARSER = ON
    DEAL_II_TRILINOS_WITH_SACADO = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_MUPARSER = ${DEAL_II_WITH_MUPARSER}
    DEAL_II_TRILINOS_WITH_SACADO = ${DEAL_II_TRILINOS_WITH_SACADO}
which conflict with the requirements."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

message("Just to check the flags used:")
message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")