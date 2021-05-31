#ifndef _ENET_CHROMOSOME_HPP_
#define _ENET_CHROMOSOME_HPP_

#include "innovation.hpp"
#include "neural_network.hpp"
#include "random.hpp"

#include <cmath>
#include <unordered_map>
#include <vector>

#define ENET_HIGHEST_COMPATIBILITY 1
#define ENET_LOWEST_COMPATIBILITY 0
#define ENET_MAX_COMPATIBILITY_VALUE_DIFF 2

namespace enet
{
template<typename Data>
struct gene
{
  using data_type = Data;

  innovation_t innovation;
  float value;
  data_type data;
  bool disabled;
};

template<typename Data>
class chromosome
{
public:
  using gene_data_type = Data;
  using gene_type = gene<gene_data_type>;
  using chromosome_type = chromosome<Data>;
  using size_type = size_t;

public:
  class iterator;

  /**
   * @brief Returns a compatibility score in the range of 0 to 1 where 1 is the highest.
   * 
   * This is used to determine how different two chromosomes are.
   * 
   * This can be used to speciate chromosomes and restrict crossover to only those that
   * are compatible (compatibility greater than a threashold). This can have better 
   * evolutionary results because of competition in a neiche and better overall fitness
   * of the offspring because of the increased similarity.
   * 
   * This operation is O(n)
   * 
   * @param other Chromosome to test compatibility with
   * @return float Compatibility score from 0 (lowest) to 1 (highest)
   */
  float compatibility(const chromosome_type& other) const
  {
    const float total = static_cast<float>(std::max(size(), other.size()));

    // If there are no genes in both chromosomes then the genes are identical
    // (no genes = no genes), so we can just return the highest compatibility.
    // This avoids needing to handle special cases for zero division later on.
    if (total == 0) return ENET_HIGHEST_COMPATIBILITY;

    unsigned int matching = 0;
    float value_diff_sum = 0;

    int i = 0, j = 0;

    while (i < size() && j < other.size())
    {
      auto& g1 = _genes[i];
      auto& g2 = other._genes[j];

      if (g1.innovation == g2.innovation)
      {
        // This is capped per gene to allows us to have a fixed compatibility range.
        // Also eliminates the case where a single gene can have a huge diff.
        // We will normalize the value diff later on to avoid doing extra calculations.
        value_diff_sum += std::fmin(std::fabs(g1.value - g2.value), ENET_MAX_COMPATIBILITY_VALUE_DIFF);

        matching++;
        i++;
        j++;
      }
      else if (g1.innovation > g2.innovation)
        j++;
      else
        i++;
    }

    // No point in doing calculations if there are no matching genes since
    // this would always result in lowest compatibility.
    // This avoids needing to handle special cases for zero division later on.
    if (matching == 0) return ENET_LOWEST_COMPATIBILITY;

    // Normalize value diff here to avoid a division at each iteration.
    value_diff_sum /= ENET_MAX_COMPATIBILITY_VALUE_DIFF;

    const float historical_ratio = matching / total;
    const float value_diff = 1 - (value_diff_sum / matching);

    // Both the historical ratio and value diff can range from 0-1. This means that
    // a multiplication of both will always result in a value range of 0-1.
    // This way of calculating compatibility is significantly different from
    // the proposed way in the neat paper.
    return historical_ratio * value_diff;
  }

  /**
   * @brief Returns an offspring of the this chromosome and the provided one.
   * 
   * Mixes up the genes and creates a new chromosome with a randomization of
   * the common gene innovation from both parents with all the excess innovations
   * of the first parent.
   * 
   * @note In theory, the less fit chromosome should be the argument of this method
   * (second parent).
   * 
   * Gene values, data and disabled state are all randomized individually,
   * so genes are not guarented to resemble the gene of one of the parents.
   * They can partially resemble it.
   * 
   * This operation is O(n).
   * 
   * @warning The chromosome given as argument is the second parent and cannot give 
   * any unique innovations to the offspring.
   * 
   * @param other 
   * @return chromosome_type 
   */
  chromosome_type crossover(const chromosome_type& other) const
  {
    auto& rand = tl_rand();

    // Start with a copy of parent 1 (this).
    // Will use parent one as the master chromosome. Only the genes that are
    // common with parent 1 can be given to the offspring.
    chromosome_type offspring { *this };

    for (size_t i = 0, j = 0; i < offspring.size() && j < other.size();)
    {
      auto& g1 = offspring._genes[i];
      auto& g2 = other._genes[j];

      if (g1.innovation == g2.innovation)
      {
        // We can generate one random state and use the 32 bits to obtains up to
        // 32 random bools.
        const uint32_t random_value = rand.next_uint32();

        // Randomize all gene fields individual instead of swaping genes
        // to obtains more randomness
        if (extract_bool(random_value, 0)) g1.value = g2.value;
        if (extract_bool(random_value, 1)) g1.data = g2.data;
        if (extract_bool(random_value, 2)) g1.disabled = g2.disabled;

        i++;
        j++;
      }
      else if (g1.innovation > g2.innovation)
        j++;
      else
        i++;
    }

    return offspring;
  }

  /**
   * @brief Adds a new gene innovation to the chromosome.
   * 
   * All new genes are enabled by default.
   * 
   * This operation is O(1).
   * 
   * @param value The master value of the gene
   * @param data The extra data for the gene
   * @return innovation_t The innovation id of the newly added gene
   */
  innovation_t extend(const float value, const gene_data_type& data)
  {
    innovation_t innovation = innovate();

    _genes.push_back({ innovation, value, data, false });

    return innovation;
  }

  /**
   * @brief Returns the iterator state for gene with the innovation to find.
   * 
   * Returns the end iterator if no gene was found.
   * 
   * Does a binary search on genes. Usually very fast.
   * 
   * Sometimes you can specify a custom low and high bound to increase performance.
   * 
   * This operation is O(log(n)).
   * 
   * @param innovation The innovation id of the gene to find
   * @param low The low bound of the search
   * @param high The high bound of the search
   * @return iterator Found iterator or end if none are found.
   */
  iterator find(const innovation_t innovation, int low, int high)
  {
    while (high >= low)
    {
      const size_type mid = low + ((high - low) >> 1);
      const innovation_t current = _genes[mid].innovation;

      if (current == innovation) return { this, mid };
      if (current > innovation) high = mid - 1;
      else
        low = mid + 1;
    }

    return end();
  }

  /**
   * @brief Returns the iterator state for gene with the innovation to find.
   * 
   * Returns the end iterator if no gene was found.
   * 
   * Does a binary search on genes. Usually very fast.
   * 
   * This operation is O(log(n)).
   * 
   * @param innovation The innovation id of the gene to find
   * @return iterator Found iterator or end if none are found.
   */
  iterator find(const innovation_t innovation)
  {
    return find(innovation, 0, _genes.size() - 1);
  }

  /**
   * @brief Returns the most recent innovation.
   * 
   * The most recent is also the highest.
   * 
   * This operation is O(1).
   * 
   * @return innovation_t Most recent innovation
   */
  innovation_t most_recent_innovation() const
  {
    return _genes.empty() ? 0 : _genes[_genes.size() - 1].innovation;
  }

  const gene_type& operator[](size_t i) const
  {
    return _genes[i];
  }

  iterator begin() { return { this, 0 }; }

  iterator end() { return { this, _genes.size() }; }

  size_type size() const { return _genes.size(); }

  bool empty() const { return _genes.empty(); }

protected:
  std::vector<gene_type> _genes;
};

template<typename Data>
class chromosome<Data>::iterator final
{
public:
  using iterator_category = std::random_access_iterator_tag;

  iterator(chromosome<Data>* const ptr, const size_type pos)
    : _ptr { ptr }, _pos { pos }
  {}

  // clang-format off
  iterator& operator+=(const size_type value) { _pos += value; return *this; }
  iterator& operator-=(const size_type value) { _pos -= value; return *this; }
  // clang-format on

  iterator& operator++() { return ++_pos, *this; }
  iterator& operator--() { return --_pos, *this; }

  iterator operator+(const size_type value) const { return { _ptr, _pos + value }; }
  iterator operator-(const size_type value) const { return { _ptr, _pos - value }; }

  bool operator==(const iterator other) const { return other._pos == _pos; }
  bool operator!=(const iterator other) const { return other._pos != _pos; }

  bool operator<(const iterator other) const { return other._pos < _pos; }
  bool operator>(const iterator other) const { return other._pos > _pos; }

  bool operator<=(const iterator other) const { return other._pos <= _pos; }
  bool operator>=(const iterator other) const { return other._pos >= _pos; }

  [[nodiscard]] const gene_type& operator*() const
  {
    return _ptr->_genes[_pos];
  }

  [[nodiscard]] gene_type& operator*()
  {
    return const_cast<gene_type&>(const_cast<const iterator*>(this)->operator*());
  }

  [[nodiscard]] const gene_type* operator->() const
  {
    return &(_ptr->_genes[_pos]);
  }

  [[nodiscard]] gene_type* operator->()
  {
    return const_cast<gene_type*>(const_cast<const iterator*>(this)->operator->());
  }

private:
  friend chromosome<Data>;

  chromosome<Data>* const _ptr;
  size_type _pos;
};

struct brain_gene_data
{
  uint32_t source;
  uint32_t target;
};

enum neuron_type
{
  SENSOR,
  OUTPUT,
  HIDDEN
};

class brain_chromosome : public chromosome<brain_gene_data>
{
public:
  struct neuron
  {
    float depth;
    neuron_type type;
    std::vector<uint32_t> incomming;
    std::vector<uint32_t> outgoing;
  };

public:
  brain_chromosome()
    : _sensors(0), _outputs(0)
  {}

  neural_network build_network() const
  {
    neural_network network { _neurons.size(), _genes.size(), _sensors, _outputs };

    std::vector<size_t> order = t_sort();

    size_t step_counter = 0, input_counter = 0, output_counter = 0;

    for (size_t i = 0; i < _neurons.size(); i++)
    {
      size_t next = order[_neurons.size() - 1 - i];

      const neuron& neuron = _neurons[next];

      network.mapping(next) = i;

      for (auto it = neuron.incomming.begin(); it != neuron.incomming.end(); ++it)
      {
        const gene_type& gene = _genes[*it];

        if (!gene.disabled)
        {
          network.steps[step_counter++] = { gene.data.source, gene.value };
        }
      }

      network.nodes[i] = { 0, static_cast<uint32_t>(neuron.incomming.size()) };

      if (neuron.type == neuron_type::SENSOR)
      {
        network.input_mapping(input_counter++) = static_cast<uint32_t>(i);
      }
      else if (neuron.type == neuron_type::OUTPUT)
      {
        network.output_mapping(output_counter++) = static_cast<uint32_t>(i);
      }
    }

    return network;
  }

  void add_synapse(float weight, uint32_t source_neuron, uint32_t target_neuron)
  {
    extend(weight, { source_neuron, target_neuron });

    uint32_t index = static_cast<uint32_t>(_genes.size() - 1);

    _neurons[source_neuron].outgoing.push_back(index);
    _neurons[target_neuron].incomming.push_back(index);
  }

  void add_neuron(float depth, neuron_type type)
  {
    if (type == SENSOR) _sensors++;
    if (type == OUTPUT) _outputs++;

    _neurons.push_back({ depth, type });
  }

  const neuron& get_neuron(size_t neuron) const
  {
    return _neurons[neuron];
  }

  size_type neuron_count() const { return _neurons.size(); }

  size_type sensors() const { return _sensors; }

  size_type outputs() const { return _outputs; }

private:
  std::vector<size_t> t_sort() const
  {
    std::vector<size_t> stack;

    stack.reserve(_neurons.size());

    bool* visited = new bool[_neurons.size()] {};

    for (size_t i = 0; i < _neurons.size(); i++)
    {
      if (!visited[i]) visit(i, stack, visited);
    }

    delete[] visited;

    return stack;
  }

  void visit(size_t i, std::vector<size_t>& stack, bool* visited) const
  {
    visited[i] = true;

    const neuron& neuron = _neurons[i];

    for (auto it = neuron.outgoing.begin(); it != neuron.outgoing.end(); ++it)
    {
      const gene_type& gene = _genes[*it];

      if (!gene.disabled && !visited[gene.data.target])
      {
        visit(gene.data.target, stack, visited);
      }
    }

    stack.push_back(i);
  }

private:
  std::vector<neuron> _neurons;
  size_t _sensors;
  size_t _outputs;
};

/**
 * @brief Utility method for creating fully connected brain chromosomes.
 * 
 * Usually used for testing purposes.
 * 
 * The first layer is the input layer, the last is the output layer and everything
 * in between is a hidden layer.
 * 
 * @warning Must have atleast 2 layers (1 input & output layer).
 * 
 * @param layers Vector of neurons per layer.
 * @param random_weights True if the weights should be randomized between -2 to 2, otherwise always 1
 * @return brain_chromosome Fully connected brain chromosome.
 */
inline brain_chromosome make_fully_connected(const std::vector<size_t>& layers, bool random_weights = false)
{
  brain_chromosome chromosome;

  for (size_t i = 0; i < layers[0]; i++)
  {
    chromosome.add_neuron(0, neuron_type::SENSOR);
  }

  for (size_t i = 1; i < layers.size() - 1; i++)
  {
    float depth = i / (layers.size() - 1.0f);

    for (size_t j = 0; j < layers[i]; j++)
    {
      chromosome.add_neuron(depth, neuron_type::HIDDEN);
    }
  }

  for (size_t i = 0; i < layers[layers.size() - 1]; i++)
  {
    chromosome.add_neuron(1, neuron_type::OUTPUT);
  }

  for (size_t layer = 0, counter = 0; layer < layers.size() - 1; layer++)
  {
    size_t size = layers[layer];

    for (size_t i = 0; i < size; i++)
    {
      size_t source = counter + i;

      for (size_t j = 0; j < layers[layer + 1]; j++)
      {
        size_t target = size + counter + j;

        float weight = random_weights ? tl_rand().next_float(-2, 2) : 1;

        chromosome.add_synapse(weight, static_cast<uint32_t>(source), static_cast<uint32_t>(target));
      }
    }

    counter += size;
  }

  return chromosome;
}

} // namespace enet

#endif