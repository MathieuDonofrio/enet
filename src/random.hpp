#ifndef _ENET_RANDOM_HPP_
#define _ENET_RANDOM_HPP_

#include <atomic>
#include <cstdint>
#include <ctime>
#include <limits>

namespace enet
{
/**
 * @brief Pseudo-random number generator from PCG family.
 * 
 * The PCG family are simple fast space-efficient statistically 
 * good algorithms for number generation.
 * 
 * This implmentation is based on a minimal c implementation:
 * *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
 * Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
 */
class random
{
private:
  static constexpr uint64_t multiplier = 6364136223846793005ull;
  static constexpr uint64_t increment = 1442695040888963407ull;
  static constexpr uint64_t scramble = 0x5DEECE66D;

public:
  constexpr random(const uint64_t seed)
    : _state(0)
  {
    initialize(seed);
  }

  /**
   * @brief Random floating point uniform between lower_bound and upper_bound
   * 
   * @note If the lower bound and upper bound are know at compile time and could be 
   * unsigned intergets prefer the compile time variant of this method (May be difference
   * in precision).
   * 
   * @param lower_bound Lower bound
   * @param upper_bound Upper bound
   * @return constexpr float Random uniform
   */
  [[nodiscard]] constexpr float next_float(const float lower_bound, const float upper_bound)
  {
    return lower_bound + (upper_bound - lower_bound) * next() / std::numeric_limits<uint32_t>::max();
  }

  /**
   * @brief Random floating point uniform between lower_bound and upper_bound
   * 
   * @note If the upper bound are know at compile time prefer the compile time
   * variant of this method (May be difference in precision).
   * 
   * @param lower_bound Lower bound
   * @param upper_bound Upper bound
   * @return constexpr float Random uniform
   */
  [[nodiscard]] constexpr float next_float(const float upper_bound)
  {
    return next() * (upper_bound / std::numeric_limits<uint32_t>::max());
  }

  /**
   * @brief Random floating point uniform between 0 and 1
   * 
   * @param lower_bound Lower bound
   * @param upper_bound Upper bound
   * @return constexpr float Random uniform
   */
  [[nodiscard]] constexpr float next_float()
  {
    return next() / static_cast<float>(std::numeric_limits<uint32_t>::max());
  }

  /**
   * @brief Random uniform between lower_bound and upper_bound
   * 
   * @param lower_bound Lower bound
   * @param upper_bound Upper bound
   * @return constexpr Random uniform
   */
  [[nodiscard]] constexpr uint32_t next_uint32(const uint32_t lower_bound, const uint32_t upper_bound)
  {
    return next_bounded(upper_bound - lower_bound) + lower_bound;
  }

  /**
   * @brief Random uniform between 0 and upper_bound
   * 
   * @param upper_bound Upper bound
   * @return constexpr Random uniform
   */
  [[nodiscard]] constexpr uint32_t next_uint32(const uint32_t upper_bound)
  {
    return next_bounded(upper_bound);
  }

  /**
   * @brief Random uniform
   * 
   * @return constexpr Random uniform
   */
  [[nodiscard]] constexpr uint32_t next_uint32()
  {
    return next();
  }

  /**
   * @brief Returns the current state.
   * 
   * @return constexpr uint64_t Current state
   */
  [[nodiscard]] constexpr uint64_t state() const { return _state; }

private:
  /**
   * @brief Generates a uniformaly distributed 32-bit random number 
   * smaller than specified bound.
   * 
   * The generated uniform will be bigger or equal to 0 and smaller than the bound.
   * 
   * @note Advances internal state.
   * 
   * @warning Cannot use a bound smaller than 1.
   * 
   * @note From pcg-random.org 
   * 
   * @param bound Max value exclusive
   * @return uint32_t Next uniformaly distributed 32-bit random number, bounded.
   */
  constexpr uint32_t next_bounded(const uint32_t bound)
  {
    // To avoid bias, we need to make the range of the RNG a multiple of
    // bound, which we do by dropping output less than a threshold.
    const uint32_t threshold = (-bound) % bound;

    // 82.25% of the time, we can expect it to require just one iteration
    while (true)
    {
      const uint32_t r = next();
      if (r >= threshold)
        return r % bound;
    }
  }

  /**
   * @brief Generates a uniformaly distributed 32-bit random number.
   * 
   * @note Advances the internal state.
   * 
   * @note From pcg-random.org 
   * 
   * @return uint32_t Next uniformaly distributed 32-bit random number.
   */
  constexpr uint32_t next()
  {
    uint64_t oldstate = _state;

    // Advance internal state
    _state = oldstate * multiplier + increment;

    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

  /**
   * @brief Initializes the internal state
   * 
   * Not the standard way of doing it, but works well enough and is fast.
   * 
   * Does one xor scramble on the seed and two naked internal state advancements.
   */
  constexpr void initialize(const uint64_t seed)
  {
    _state = (seed ^ scramble) * multiplier + increment;
    _state = _state * multiplier + increment;
  }

private:
  uint64_t _state;
};

/**
 * @brief Thread-safe non-deterministic unique seed.
 * 
 * Combination of time based non-determinism and atomic linear congruential generator.
 * The result is a pretty good non deterministic seed.
 *
 * Because obtaining current time is slow, it only obtains time once.
 * 
 * @return uint64_t Seed
 */
inline uint64_t seed()
{
  static constexpr uint64_t multiplier = 2862933555777941757ull;
  static constexpr uint64_t increment = 3037000493ull;

  /**
   * @brief Obtains current time and generated a time based seed.
   * 
   * Spreads the bits of the time for a better distribution. This is done because the
   * lower bits of time change faster.
   * 
   * Used as a inner function.
   */
  struct time_seed
  {
    time_seed() : time(std::time(nullptr)), seed((time << 32 | time >> 32) ^ time) {}
    uint64_t time, seed;
  };

  // Initialized with time based seed and one iteration of generator
  static std::atomic<uint64_t> uniquifier = time_seed().seed * multiplier + increment;

  while (true)
  {
    uint64_t current = uniquifier;
    uint64_t next = current * multiplier + increment;

    if (current == uniquifier.exchange(next)) return next;
  }
}

/**
 * @brief Returns the thread local instance of random.
 * 
 * If you need to obtain a random instance at runtime use this.
 *
 * Every instance is correctly seeded with a non-deterministic unique seed.
 * 
 * @return random Thread local random instance
 */
inline random& tl_rand()
{
  static thread_local random local_random { seed() };

  return local_random;
}

/**
 * @brief Returns a bool from a value.
 * 
 * Every bit in a value can represent a boolean value. That means in a 
 * 32 bit integer there can be 32 booleans extracted.
 * 
 * We can use this to generate 32 random bools from one state.
 * 
 * @warning Depth ranges from 0 to 31 where 31 is 32nd bit. Using a depth of more than 31
 * results in undefined behaviour.
 * 
 * @param value The 32 bit value to extract the bool from
 * @param bit The bit to extract the bool from (max == 31)
 * 
 * @return Boolean state of bit for depth in value
 */
inline constexpr bool extract_bool(uint32_t value, uint32_t bit = 0)
{
  return value & (1 << bit);
}

} // namespace enet

#endif