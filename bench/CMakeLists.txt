#
# Bench
#

add_executable(benchmarks benchmark.cpp)
target_link_libraries(benchmarks PRIVATE ENET)
add_test(NAME benchmarks COMMAND benchmarks)