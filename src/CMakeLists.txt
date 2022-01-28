add_library(mcmcsampler
    gaussian_proposal.hpp
    ${PROJECT_SOURCE_DIR}/include/mcmcsampler/dram.hpp
    dram.cpp
    ${PROJECT_SOURCE_DIR}/include/mcmcsampler/mh.hpp
    ${PROJECT_SOURCE_DIR}/include/mcmcsampler/am.hpp
    ${PROJECT_SOURCE_DIR}/include/mcmcsampler/dr.hpp
    dr.cpp
)
target_include_directories(mcmcsampler PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(mcmcsampler PUBLIC
    Eigen3::Eigen
)
target_compile_features(mcmcsampler PUBLIC
    cxx_std_11
)

set_target_properties(mcmcsampler PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
