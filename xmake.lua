add_rules("mode.debug", "mode.release")

-- 添加 CUDA 和 Google Test 支持
add_requires("cuda", {configs = {shared = false}})
add_requires("gtest", {configs = {shared = false}})
add_requires("glog", {configs = {shared = false}})
add_requires("sentencepiece", {configs = {shared = false}})
add_requires("armadillo", {configs = {shared = false}})
add_includedirs("include")
add_includedirs("include/base")
add_includedirs("include/model")
add_includedirs("include/op")
add_includedirs("source/op/kernels")
add_includedirs("source/op/kernels/cpu")
add_includedirs("source/op/kernels/cuda")
add_includedirs("include/sampler")
add_includedirs("include/tensor")
add_rules("cuda")

add_defines("ARMA_ALLOW_FAKE_GCC")
-- 定义静态库 target
target("libinfer")
    set_kind("static")
    set_languages("c++17")
    add_files("source/base/*.cpp") 
    add_files("source/model/*.cpp")
    add_files("source/op/*.cpp")
    add_files("source/sampler/*.cpp")
    add_files("source/tensor/*.cpp")
    add_files("source/op/kernels/cpu/*.cpp")
    add_files("source/op/kernels/*.cpp")
    add_files("source/op/kernels/cuda/*.cu")
    
    add_packages("cuda","glog", "sentencepiece","armadillo")
    add_links("pthread")  -- 添加 pthread 链接
    add_syslinks("pthread", "dl", "rt") -- 添加必要的系统库链接
    set_policy("build.cuda.devlink", true) 
-- 定义 infer 可执行文件 target
target("infer")
    set_kind("binary")
    set_languages("c++17")
    add_files("source/main.cpp")
    
    add_deps("libinfer")
    add_packages("cuda")
    add_links("pthread")

target("inferdemo")
    set_kind("binary")
    set_languages("c++17")
    add_files("demo/main.cpp")
    
    add_deps("libinfer")
    add_packages("cuda","glog","armadillo", "sentencepiece")
    add_links("pthread")


target("test")
    set_kind("binary")
    set_languages("c++17")
    add_files("test/*.cpp")
    add_files("test/*.cu")  -- 添加这行以包含 CUDA 文件
    add_files("test/test_cu/*.cpp")
    add_files("test/test_model/*.cpp")
    add_files("test/test_op/*.cpp")
    add_files("test/test_tensor/*.cpp")
    add_files("test/optimized/*.cpp")
    add_includedirs("test")
    add_deps("libinfer")
    add_packages("gtest", "cuda","glog", "sentencepiece","armadillo")
    add_links("gtest", "pthread", "cuda")