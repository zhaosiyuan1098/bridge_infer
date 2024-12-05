add_rules("mode.debug", "mode.release")

-- 添加 CUDA 和 Google Test 支持
add_requires("cuda", "gtest","glog","sentencepiece","armadillo")
add_includedirs("include")

-- 定义静态库 target
target("libinfer")
    set_kind("shared")
    set_languages("c++20")
    add_files("source/base/*.cpp") 
    add_files("source/model/*.cpp")
    add_files("source/op/*.cpp")
    add_files("source/sampler/*.cpp")
    add_files("source/tensor/*.cpp")
    add_includedirs("include/base")
    add_includedirs("include/model")
    add_includedirs("include/op")
    add_includedirs("include/sampler")
    add_includedirs("include/tensor")
    add_packages("cuda","glog", "sentencepiece","armadillo")
    add_links("pthread")  -- 添加 pthread 链接
    add_syslinks("pthread", "dl", "rt") -- 添加必要的系统库链接

-- 定义 infer 可执行文件 target
target("infer")
    set_kind("binary")
    set_languages("c++20")
    add_files("source/main.cpp")
    add_includedirs("include")
    add_deps("libinfer")
    add_packages("cuda")
    add_links("pthread")

-- -- 定义 tests 可执行文件 target
-- target("tests")
--     set_kind("binary")
--     set_languages("c++20")
--     add_files("test/*.cpp")
--     add_includedirs("include")
--     add_deps("libinfer")
--     add_packages("gtest", "cuda")
--     add_links("gtest", "pthread", "cuda")