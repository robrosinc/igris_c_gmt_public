cmake_minimum_required(VERSION 3.16)
project(public_inference_module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_link_options("-Wl,--no-as-needed")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(igris_c_sdk REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
unset(ONNXRUNTIME_LIBRARY CACHE)

find_path(
  ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_c_api.h
  HINTS /usr/local/include /usr/include
  PATH_SUFFIXES onnxruntime
)

find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime libonnxruntime
  HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(
    FATAL_ERROR
      "ONNX Runtime not found. Set ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY if installed in a custom location."
  )
endif()

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(
  onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

if(TARGET yaml-cpp::yaml-cpp)
  set(PUBLIC_INFERENCE_YAML_TARGET yaml-cpp::yaml-cpp)
else()
  set(PUBLIC_INFERENCE_YAML_TARGET yaml-cpp)
endif()

add_library(
  public_inference_module_core
  src/config.cpp
  src/inference_module.cpp
  src/motion_frame_source.cpp
  src/ros_motion_receiver.cpp
  src/robot_io.cpp
)

target_include_directories(
  public_inference_module_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
)

target_link_libraries(
  public_inference_module_core
  PUBLIC Eigen3::Eigen
         igris_c_sdk::igris_c_sdk
         rclcpp::rclcpp
         ${std_msgs_TARGETS}
         ${PUBLIC_INFERENCE_YAML_TARGET}
         onnxruntime::onnxruntime
)

add_executable(public_inference_module_node src/main.cpp)
target_link_libraries(public_inference_module_node PRIVATE public_inference_module_core)

install(
  TARGETS public_inference_module_core public_inference_module_node
  EXPORT public_inference_module-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY data/ DESTINATION share/${PROJECT_NAME}/data)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

install(
  EXPORT public_inference_module-targets
  FILE public_inference_module-targets.cmake
  NAMESPACE public_inference_module::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
