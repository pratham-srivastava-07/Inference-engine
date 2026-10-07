# Each test is a plain executable: exit code 0 = pass, non-zero = fail.
# No external test framework is used.

add_executable(test_tensor
    test_tensor.cpp
)

target_link_libraries(test_tensor PRIVATE mini_infer)
mini_infer_set_project_options(test_tensor)

add_test(NAME tensor.scaffold COMMAND test_tensor)
