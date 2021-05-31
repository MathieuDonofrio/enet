#include <random.hpp>
#include <gtest/gtest.h>

using namespace enet;

TEST(Random, Uint32_UpperBound_InRange)
{
    for(size_t seed = 0; seed < 100; seed++)
    {
        enet::random r { seed };

        for(size_t i = 0; i < 10000; i++)
        {
            auto upper_bound = 1 + seed * 3;

            auto value = r.next_uint32(upper_bound);

            ASSERT_GE(value, 0);
            ASSERT_LE(value, upper_bound);
        }
    }
}

TEST(Random, Uint32_LowerBoundAndUpperBound_InRange)
{
    for(size_t seed = 0; seed < 100; seed++)
    {
        enet::random r { seed };

        for(size_t i = 0; i < 10000; i++)
        {
            auto lower_bound = seed;
            auto upper_bound = 1 + seed * 3;

            auto value = r.next_uint32(lower_bound, upper_bound);

            ASSERT_GE(value, lower_bound);
            ASSERT_LE(value, upper_bound);
        }
    }
}

TEST(Random, NextFloat_01_InRange)
{
    for(size_t seed = 0; seed < 100; seed++)
    {
        enet::random r { seed };

        for(size_t i = 0; i < 10000; i++)
        {
            auto value = r.next_float();

            ASSERT_GE(value, 0);
            ASSERT_LE(value, 1);
        }
    }
}

TEST(Random, NextFloat_UpperBound_InRange)
{
    for(size_t seed = 0; seed < 100; seed++)
    {
        enet::random r { seed };

        for(size_t i = 0; i < 10000; i++)
        {
            auto upper_bound = 1 + seed * 3;

            auto value = r.next_float(upper_bound);

            ASSERT_GE(value, 0);
            ASSERT_LE(value, upper_bound);
        }
    }
}

TEST(Random, NextFloat_LowerBoundAndUpperBound_InRange)
{
    for(size_t seed = 0; seed < 100; seed++)
    {
        enet::random r { seed };

        for(size_t i = 0; i < 10000; i++)
        {
            auto lower_bound = seed;
            auto upper_bound = 1 + seed * 3;

            auto value = r.next_float(lower_bound, upper_bound);

            ASSERT_GE(value, lower_bound);
            ASSERT_LE(value, upper_bound);
        }
    }
}