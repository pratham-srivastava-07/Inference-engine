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

add_executable(test_relu test_relu.cpp)
target_link_libraries(test_relu PRIVATE mini_infer)
mini_infer_set_project_options(test_relu)
add_test(NAME relu.test_relu COMMAND test_relu)

add_executable(test_softmax test_softmax.cpp)
target_link_libraries(test_softmax PRIVATE mini_infer)
mini_infer_set_project_options(test_softmax)
add_test(NAME softmax.test_softmax COMMAND test_softmax)

add_executable(test_layer_norm test_layer_norm.cpp)
target_link_libraries(test_layer_norm PRIVATE mini_infer)
mini_infer_set_project_options(test_layer_norm)
add_test(NAME layerNorm.test_layer_norm COMMAND test_layer_norm)