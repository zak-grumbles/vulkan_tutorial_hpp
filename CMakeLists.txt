cmake_minimum_required(VERSION 3.12)
project(VulkanTutorialHPP)

# Find Vulkan
find_package(Vulkan REQUIRED FATAL_ERROR)

# GLFW
if(WIN32)
	set(glfw3_DIR "C:\\Program Files (x86)\\GLFW\\lib\\cmake\\glfw3")
endif()
find_package(glfw3 REQUIRED FATAL_ERROR)

# Source files
set(VK_HPP_SRC 
    main.cpp
	app.h
	app.cpp
)

# Executable
add_executable(vk_hpp ${VK_HPP_SRC})

# Link libs
target_include_directories(vk_hpp PUBLIC ${GLFW_INCLUDE_DIRS})
target_link_libraries(vk_hpp PUBLIC glfw)

target_include_directories(vk_hpp PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vk_hpp PUBLIC ${Vulkan_LIBRARIES})

# C++ Standard
set_property(TARGET vk_hpp PROPERTY CXX_STANDARD 17)

# Compiler options
target_compile_options(vk_hpp PRIVATE -g)