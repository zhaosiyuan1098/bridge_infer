#include "math_utlis.h"
#include <gtest/gtest.h>

TEST(MathUtilsTest, Subtract) {
    EXPECT_EQ(math_utils::subtract(2, 1), 1);
    EXPECT_EQ(math_utils::subtract(1, 1), 0);
    EXPECT_EQ(math_utils::subtract(-1, -1), 0);
}