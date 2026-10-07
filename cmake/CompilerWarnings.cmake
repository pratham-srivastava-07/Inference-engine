# Per-target compiler settings.
#
# Instead of setting global flags (CMAKE_CXX_FLAGS / add_compile_options),
# each target opts in by calling this function. The options are PRIVATE, so
# they apply only to the target's own sources and never leak to consumers.

function(mini_infer_set_project_options target)
    # Use -std=c++20 rather than -std=gnu++20 (no compiler extensions).
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
        )
    endif()
endfunction()
