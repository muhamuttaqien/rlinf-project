# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Hashes pin the inspected implementation; patch only a disposable build copy.
set(vendor_dir "${CMAKE_CURRENT_BINARY_DIR}/vendor")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/observation.patch")
function(copy_audited relative expected)
  set(source "${ROBOTIQ_SOURCE_DIR}/${relative}")
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "Missing audited Robotiq source: ${source}")
  endif()
  file(SHA256 "${source}" actual)
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "Robotiq source changed: ${relative}: ${actual}; review before updating hashes")
  endif()
  get_filename_component(directory "${vendor_dir}/${relative}" DIRECTORY)
  file(MAKE_DIRECTORY "${directory}")
  configure_file("${source}" "${vendor_dir}/${relative}" COPYONLY)
endfunction()
copy_audited(include/robotiq/constants.h 75401d10d9be9688a3bfa0e5f0630317c6e26cce415b4dc20d4f6ebc10fc2642)
copy_audited(include/robotiq/types.h 335133bd6a4d86ecf965c6ce4de29983228de65250d02322eea98ed6a7df5c6a)
copy_audited(include/robotiq/robotiq_gripper_interface.h 3b944cf80ffc47a1e2079128d811f4dd2871b64d59ac8a5251a9eaf27335ef83)
copy_audited(src/robotiq_gripper_interface.cc f156ab602ba9d766cdb4812ebd93bbb6e133365feea8c84c54d586798b7c9979)
copy_audited(src/helpers.cc 2a212b59cd8be97d8c722439b0a83eea9900ac18747b210b5313742a9cf17570)
copy_audited(src/helpers.h c5116a31376ba80a71367341b0eb0c44851290483f45857101d9041edc99ea79)
copy_audited(src/timeout_reader.cc dc2b2369e2b2f0e58595771968b5baeb0be643d0122fbfbd3f5bdceecc2595d3)
copy_audited(src/timeout_reader.h 53a156bb9e69deb20f2c00952103acd67802f27ca26e27d6704274bbc5cbd87d)
find_program(PATCH_EXECUTABLE patch REQUIRED)
execute_process(COMMAND "${PATCH_EXECUTABLE}" --batch --fuzz=0 -p1 -i
  "${CMAKE_CURRENT_LIST_DIR}/observation.patch"
  WORKING_DIRECTORY "${vendor_dir}" RESULT_VARIABLE patch_result OUTPUT_VARIABLE patch_log ERROR_VARIABLE patch_error)
if(NOT patch_result EQUAL 0)
  message(FATAL_ERROR "Observation patch failed: ${patch_log} ${patch_error}")
endif()
