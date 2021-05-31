#ifndef _ENET_INNOVATION_HPP_
#define _ENET_INNOVATION_HPP_

#include <atomic>
#include <cstdint>

namespace enet
{
// Surpassing the maximum possible value will result in undefined behaviour,
// it is best to always use 64 bits unless you know what your doing.
using innovation_t = uint64_t;

namespace internal
{
  /**
   * @brief Atomic counter for thread safe unique id
   * 
   * Starts at zero but will skip it, because 0 is not a valid innovation.
   */
  static std::atomic<innovation_t> counter = 0;
} // namespace internal

/**
 * @brief Generated a new innovation.
 * 
 * @note Thread-safe.
 * 
 * @return innovation_t Innovation
 */
inline innovation_t innovate() { return internal::counter.fetch_add(1); }

/**
 * @brief Returns the current innovation.
 * 
 * This would be the next innovation.
 * 
 * @warning Does not create a new innovation.
 * 
 * @return innovation_t Current innovation
 */
inline innovation_t read_innovation() { return internal::counter.load() + 1; }

/**
 * @brief Sets the current innovation to specified value.
 * 
 * This would be the next innovation.
 * 
 * Used to resume a simulation.
 * 
 * @param innovation The innovation value to set.
 */
inline void write_innovation(innovation_t innovation) { return internal::counter.store(innovation - 1); }
} // namespace enet

#endif
