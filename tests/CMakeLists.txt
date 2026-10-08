add_executable(test_matmul matrix_multiplication.cpp)

target_link_libraries(test_matmul PRIVATE mini_infer)

add_test(NAME tensor.matmul COMMAND test_matmul)