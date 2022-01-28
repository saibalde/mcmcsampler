add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

mark_as_advanced(
    BUILD_GTEST
    BUILD_GMOCK
    BUILD_SHARED_LIBS
    gtest_build_samples
    gtest_build_tests
    gtest_disable_pthreads
    gtest_force_shared_crt
    gtest_hide_internal_symbols
    gmock_build_tests
)

set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

macro(mcmcsampler_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} PRIVATE mcmcsampler gtest)
    target_compile_features(${TESTNAME} PRIVATE cxx_std_11)
    gtest_discover_tests(${TESTNAME}
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()

mcmcsampler_add_test(mh_test mh_test.cpp)
mcmcsampler_add_test(am_test am_test.cpp)
mcmcsampler_add_test(dr_test dr_test.cpp)
mcmcsampler_add_test(dram_test dram_test.cpp)

configure_file(plots.py plots.py COPYONLY)
configure_file(plots.tex plots.tex COPYONLY)
