# See the following link for reasonable minimum version suggestions
# https://cliutils.gitlab.io/modern-cmake/chapters/intro/dodonot.html#what-minimum-to-choose-os-support
cmake_minimum_required(VERSION 3.20...4.1)

project(Fulqrum
    VERSION 0.0.0
)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP COMPONENTS C CXX REQUIRED)
add_compile_options(-Wall -Wextra -O3 -D_GLIBCXX_ASSERTIONS)

include_directories(fulqrum/include)
add_subdirectory(third-party/doctest)


add_executable(fulqrum_tests
    test/doctest_main.cpp
    test/test_qubit_operator.cpp
    test/test_qubit_operator_offdiag.cpp
    test/test_grouping.cpp
    test/test_ladder.cpp
    test/test_type2_table.cpp
    test/test_fermi.cpp
    test/test_jw.cpp
    test/test_diag.cpp
    test/test_convert.cpp
)
target_include_directories(fulqrum_tests PRIVATE third-party/doctest/doctest)
target_link_libraries(fulqrum_tests
    PRIVATE
)

include(third-party/doctest/scripts/cmake/doctest.cmake)
doctest_discover_tests(fulqrum_tests)

