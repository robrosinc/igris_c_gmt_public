cmake_minimum_required(VERSION 3.16)
project(igris_c_gmt_public)

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
  PATH_SUFFIXES onnxruntime)

find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime libonnxruntime
  HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(
    FATAL_ERROR
      "ONNX Runtime not found. Set ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY if installed in a custom location."
  )
endif()

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(
  onnxruntime::onnxruntime
  PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
             INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

if(TARGET yaml-cpp::yaml-cpp)
  set(IGRIS_C_GMT_PUBLIC_YAML_TARGET yaml-cpp::yaml-cpp)
else()
  set(IGRIS_C_GMT_PUBLIC_YAML_TARGET yaml-cpp)
endif()

add_library(
  igris_c_gmt_public_core
  src/core/config.cpp
  src/io/robot_io.cpp
  src/motion/frame_source.cpp
  src/motion/ros_receiver.cpp
  src/modules/action_builder.cpp
  src/modules/motion_handler.cpp
  src/modules/obs_builder.cpp
  src/modules/onnx_runner.cpp
  src/modules/state_handler.cpp)

target_include_directories(
  igris_c_gmt_public_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

target_link_libraries(
  igris_c_gmt_public_core
  PUBLIC Eigen3::Eigen igris_c_sdk::igris_c_sdk rclcpp::rclcpp
         ${std_msgs_TARGETS} ${IGRIS_C_GMT_PUBLIC_YAML_TARGET}
         onnxruntime::onnxruntime)

add_executable(igris_c_gmt_public_node src/main.cpp)
target_link_libraries(igris_c_gmt_public_node PRIVATE igris_c_gmt_public_core)

add_executable(gmr_control_mode_dds_bridge
               src/tools/gmr_control_mode_dds_bridge.cpp)
target_link_libraries(gmr_control_mode_dds_bridge
                      PRIVATE igris_c_sdk::igris_c_sdk rclcpp::rclcpp
                              ${std_msgs_TARGETS})

install(
  TARGETS igris_c_gmt_public_core igris_c_gmt_public_node
          gmr_control_mode_dds_bridge
  EXPORT igris_c_gmt_public-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY data/ DESTINATION share/${PROJECT_NAME}/data)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
install(PROGRAMS scripts/plot_inference_csv.py
        DESTINATION share/${PROJECT_NAME}/scripts)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

install(
  EXPORT igris_c_gmt_public-targets
  FILE igris_c_gmt_public-targets.cmake
  NAMESPACE igris_c_gmt_public::
  DESTINATION lib/cmake/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
