# ATT2 thin wrapper (desk 2026-10-27): builds ONLY vLLM's `_flashmla_C` target exactly the way vLLM af7f9488 does.
#
# It replays the CUDA branch of the af7f9488 top-level CMakeLists.txt prelude (lines 14-293: project, C++20, utils.cmake,
# python, torch, CUDA_SUPPORTED_ARCHS, gencode normalisation, VLLM_GPU_FLAGS, NVCC_THREADS, --compress-mode, CXX CUDA
# include flags, FetchContent) and then includes cmake/external_projects/flashmla.cmake UNCHANGED (copied byte-for-byte
# from af7f9488). Everything vLLM's top level does between the prelude and the flashmla include (the _C / _moe_C /
# cutlass / marlin sections) is skipped: it only appends -DENABLE_* / -DVLLM_ENABLE_COOPERATIVE_TOPK defines to
# VLLM_GPU_FLAGS, none of which any FlashMLA source reads (grep: 0 hits in csrc/ outside cutlass), and it FetchContent's
# CUTLASS v4.7.1 which _flashmla_C does not use (it includes FlashMLA's own csrc/cutlass submodule @147f5673).
#
# Configure (see build_flashmla_splitkv.sh.DRAFT):
#   TORCH_CUDA_ARCH_LIST="9.0 10.0" cmake -S <this dir> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
#     -DVLLM_PYTHON_EXECUTABLE=$(which python3) -DFLASH_MLA_SRC_DIR=<patched FlashMLA> -DFETCHCONTENT_BASE_DIR=<dir> \
#     -DNVCC_THREADS=2 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
#   cmake --build <build> --target _flashmla_C
cmake_minimum_required(VERSION 3.26)

project(vllm_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(ignoreMe "${VLLM_PYTHON_PATH}")
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13" "3.14")
set(TORCH_SUPPORTED_VERSION_CUDA "2.13.0")

if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR "Please set VLLM_PYTHON_EXECUTABLE")
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

find_package(Torch REQUIRED)

if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.4)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.7;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

if (NOT HIP_FOUND AND NOT PYTORCH_FOUND_HIP AND CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")
  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
else()
  message(FATAL_ERROR "Can't find CUDA installation (CUDA_FOUND false after find_package(Torch)).")
endif()

clear_cuda_gencode_flags(CUDA_ARCH_FLAGS)
warn_if_ptx_arch_requested("${CUDA_ARCH_FLAGS}")
extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
cuda_archs_loose_intersection(CUDA_ARCHS "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
if(NOT CUDA_ARCHS)
  message(FATAL_ERROR "No supported CUDA architectures (flags: ${CUDA_ARCH_FLAGS}). Set TORCH_CUDA_ARCH_LIST.")
endif()

get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA" AND
   DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND VLLM_GPU_FLAGS "--compress-mode=size")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include")
  if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
  endif()
endif()

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR})
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

message(STATUS "ATT2 VLLM_GPU_FLAGS: ${VLLM_GPU_FLAGS}")
message(STATUS "ATT2 CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "ATT2 CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# Unchanged af7f9488 file. FLASH_MLA_SRC_DIR (env or -D) points it at the PR #25 tree; no network is touched.
include(cmake/external_projects/flashmla.cmake)

# ---- ATT2b (Card W6, desk 2026-10-28): build PR #25 as a SIDE extension, never a replacement for vllm/_flashmla_C ----
# The sources register under torch library namespace _flashmla_splitkv_C (patches/01-splitkv-namespace.diff), so this
# .so loads next to the shipped vllm/_flashmla_C.abi3.so in one process (torch.ops.load_library; never imported).
# Its kernel templates / host stubs share mangled names with the shipped .so but use PR #25's Params layout, so:
#   * every non-API symbol is hidden (C++ and CUDA host code) -> nothing to interpose in either direction;
#   * -Bsymbolic: whatever stays exported binds INSIDE this .so first (DT_SYMBOLIC / DF_SYMBOLIC).
# The target keeps vLLM's internal name _flashmla_C (flashmla.cmake is verbatim); only the output file is renamed.
set_target_properties(_flashmla_C PROPERTIES
    OUTPUT_NAME _flashmla_splitkv_C
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_compile_options(_flashmla_C PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility-inlines-hidden>)
target_link_options(_flashmla_C PRIVATE "LINKER:-Bsymbolic")
message(STATUS "ATT2b: _flashmla_C target -> _flashmla_splitkv_C (hidden visibility, -Bsymbolic)")
