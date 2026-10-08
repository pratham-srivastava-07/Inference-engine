# Each test is a plain executable: exit code 0 = pass, non-zero = fail.
# No external test framework is used.

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE mini_infer)
mini_infer_set_project_options(test_tensor)
add_test(NAME tensor.scaffold COMMAND test_tensor)

add_executable(test_matmul matrix_multiplication.cpp)
target_link_libraries(test_matmul PRIVATE mini_infer)
mini_infer_set_project_options(test_matmul)
add_test(NAME tensor.matmul COMMAND test_matmul)

add_executable(test_linear test_linear.cpp)
target_link_libraries(test_linear PRIVATE mini_infer)
mini_infer_set_project_options(test_linear)
add_test(NAME linear.test_linear COMMAND test_linear)
