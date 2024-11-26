#include "math_utlis.h"
#include <gtest/gtest.h>

// 测试 add 函数
TEST(MathUtilsTest, Add) {
    EXPECT_EQ(math_utils::add(1, 1), 2);
    EXPECT_EQ(math_utils::add(-1, 1), 0);
    EXPECT_EQ(math_utils::add(-1, -1), -2);
}

