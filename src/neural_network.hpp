#ifndef _ENET_NEURAL_NETWORK_HPP_
#define _ENET_NEURAL_NETWORK_HPP_

#include <cstdint>
#include <cstring>
#include <vector>

#define MAX_ACTIVATION_ERROR 0.04f
#define AVERAGE_ACTIVATION_ERROR 0.01f

namespace enet
{
/**
 * @brief Node/Neuron in a neural network.
 * 
 * Hold the nodes current activation as well as the amount of
 * steps/dependancies it needs to compleate/compute before activating.
 * 
 * @note Activation is always mutable and is used as a cache.
 */
struct network_node
{
  mutable float activation;
  uint32_t steps;
};

/**
 * @brief Synapse/Progagation step in a neural network.
 * 
 * Hold the synapse weight aswell as the supplier/source node
 * that it need to be weighted to.
 */
struct network_step
{
  uint32_t source;
  float weight;
};

/**
 * @brief Fast approximation of sigmoid activation.
 * 
 * The approximation function is an approximation of the sigmoid function 
 * recommended in the official neat paper.
 * 
 * Approximation of: 1 / (1 + exp(-4.9 * x))
 * 
 * @note The maximum approximation error is guaranted to be lower than MAX_ACTIVATION_ERROR 
 * and the average error for the critical range of -2 to 2 is guaranted to be
 * lower than AVERAGE_ACTIVATION_ERROR.
 * 
 * @note Atleast 2 times faster than implementation with cmath exp depending on compiler.
 * With GCC you can expect 5-6 times faster.
 * 
 * @param x The value to apply activation function to
 * @return constexpr float The result
 */
constexpr float activate(const float x)
{
  // We need to do this because values exceding this range are no longer
  // sigmoid-like
  if (x <= -1) return 0;
  if (x >= 1) return 1;

  // Very similar to 1 / (1 + exp(-4.9 * x)) for x ranges between -1 and 1
  // Type this into a program like desmos to see what it looks like
  return (x / (1 + x * x)) + 0.5f;
}

/**
 * @brief Neural network for fast propagation
 * 
 * This neural network is used as a container of topologically sorted propagation steps
 * for simple and fast linear propagation threw a directed acyclic graph neural network.
 * 
 * The factory for this neural_network is reponsible for correctly initializing the network.
 * 
 * The speed comes from having all steps of propagation sorted and stored contiguiously in memory.
 * 
 * To retreive the correct locations of topologically sorted neurons, mappings for all neurons
 * are included, aswell as explicit mappings for inputs and outputs.
 * 
 * @note Also optimized for size so that is can pair well as component data for 
 * an entity-component system.
 */
struct neural_network
{
  /**
   * @brief Array that holds data about nodes/neurons in this neural network.
   */
  network_node* nodes;

  /**
   * @brief Array that holds the steps required to propagate this neural network.
   */
  network_step* steps;

  /**
   * @brief Array that holds the mappings for node locations.
   * 
   * Should be a mapping of a neuron index in the chromosome to 
   * a node index in the neural network.
   * 
   * The first mappings to the node count are neuron indexes to
   * node indexes. The next mappings up to output count are for 
   * output positions to node indexes. The final mappins up to input count
   * are for input positions to node indexes. That means that the
   * size of the mappings is equal to: node count + output count +
   * input count. 
   * 
   * @note 3 arrays have been combined into one to reduce the
   * size of the neural network by 2 pointers.
   * 
   */
  uint32_t* mappings;

  /**
   * @brief Amount of nodes in network.
   */
  uint32_t node_count;

  // Note: Step count is not nessesary because it can be calculated from nodes.

  /**
   * @brief Amount of inputs.
   * 
   * @note 16 bits should be enough.
   */
  uint16_t input_count;

  /**
   * @brief Amount of outputs.
   * 
   * @note 16 bits should be enough.
   */
  uint16_t output_count;

  /**
   * @brief Construct a new neural network object
   * 
   * @param node_count The amount of network nodes/neurons
   * @param step_count The amount network propagation steps/synapses/genes
   * @param inputs The amount of inputs/sensors
   * @param outputs The amount of outputs
   */
  neural_network(size_t node_count, size_t step_count, size_t inputs, size_t outputs)
  {
    nodes = new network_node[node_count];
    steps = new network_step[step_count];

    mappings = new uint32_t[inputs + outputs + node_count];

    this->node_count = static_cast<uint32_t>(node_count);
    input_count = static_cast<uint16_t>(inputs);
    output_count = static_cast<uint16_t>(outputs);
  }

  /**
   * @brief Copy constructor.
   * 
   * @param other Neural network to copy
   */
  neural_network(const neural_network& other)
  {
    const uint32_t step_count = other.step_count();

    nodes = new network_node[other.node_count];
    steps = new network_step[step_count];

    uint32_t mappings_count = other.node_count + other.input_count + other.output_count;
    
    mappings = new uint32_t[mappings_count];

    node_count = other.node_count;
    input_count = other.input_count;
    output_count = other.output_count;

    std::memcpy(nodes, other.nodes, other.node_count * sizeof(network_node));
    std::memcpy(steps, other.steps, step_count * sizeof(network_step));
    std::memcpy(mappings, other.mappings, mappings_count * sizeof(uint32_t));
  }

  /**
   * @brief Move constructor.
   * 
   * @param other Neural network to steal from
   */
  neural_network(neural_network&& other)
  {
    nodes = other.nodes;
    steps = other.steps;

    mappings = other.mappings;

    node_count = other.node_count;
    input_count = other.input_count;
    output_count = other.output_count;

    other.nodes = NULL;
    other.steps = NULL;
    other.mappings = NULL;
  }

  /**
   * @brief Destroy the neural network object
   */
  ~neural_network()
  {
    if (nodes) delete[] nodes;
    if (steps) delete[] steps;
    if (mappings) delete[] mappings;
  }

  /**
   * @brief Copy assignment operator.
   * 
   * @param other Neural network to copy from
   * @return neural_network& Copied neural network
   */
  neural_network& operator=(neural_network other)
  {
    swap(*this, other);

    return *this;
  }

  /**
   * @brief Move assignment operator.
   * 
   * @param other Neural network to copy from
   * @return neural_network& Copied neural network
   */
  neural_network& operator=(neural_network&& other)
  {
    swap(*this, other);

    return *this;
  }

  /**
   * @brief Swaps the data of the two neural networks.
   * 
   * Used for copy and swap idiom.
   * 
   * @param left First neural network
   * @param right Second neural network
   */
  friend void swap(neural_network& left, neural_network right)
  {
    std::swap(left.nodes, right.nodes);
    std::swap(left.steps, right.steps);
    std::swap(left.mappings, right.mappings);
    std::swap(left.node_count, right.node_count);
    std::swap(left.input_count, right.input_count);
    std::swap(left.output_count, right.output_count);
  }

  /**
   * @brief Returns the amount of steps for propagation.
   * 
   * Step count is not needed for propagation, so instead of storing it,
   * we can just recalculate it every time.
   * 
   * @return uint32_t Amount of steps for propagation
   */
  uint32_t step_count() const
  {
    uint32_t sum = 0;

    for (size_t i = 0; i < node_count; i++) sum += nodes[i].steps;

    return sum;
  }

  /**
   * @brief Returns the mapping for a node at specified position.
   * 
   * @param node The node/neuron to find activation for
   * @return network_node The network node at position
   */
  uint32_t& mapping(const size_t position) const
  {
    return mappings[position];
  }

  /**
   * @brief Returns the mapping for an input node at specified position.
   * 
   * @param node The node/neuron to find activation for
   * @return network_node The network node at position
   */
  uint32_t& input_mapping(const size_t position) const
  {
    return mappings[position + node_count];
  }

  /**
   * @brief Returns the mapping for an output node at specified position.
   * 
   * @param node The node/neuron to find activation for
   * @return network_node The network node at position
   */
  uint32_t& output_mapping(const size_t position) const
  {
    return mappings[position + node_count + input_count];
  }
};

/**
 * @brief Propagates a neural network and sets all activations for inputs.
 * 
 * Sets the input neurons to specified activation and propagates the network forward.
 * 
 * This function is extreamly fast for what it does. The complexity is O(n).
 * 
 * The outputs are stored in the activation cache of the network (in each node). There 
 * are other functions for retreiving that information.
 * 
 * @warning Using the wrong number of inputs results in undefined behaviour.
 * 
 * @warning Clears all last activations before propagating.
 * 
 * @param network The neural network to propagate
 * @param inputs The inputs to propagate in the network
 */
inline void propagate(const neural_network& network, const std::vector<float>& inputs)
{
  // Set input activations
  for (size_t i = 0; i < inputs.size(); i++)
  {
    network.nodes[network.input_mapping(i)].activation = inputs[i];
  }

  // We use embeded for loops but every step is iterated on only once.
  // For this we need to track the progress of our steps.
  uint32_t progress = 0;

  for (size_t i = 0; i < network.node_count; i++)
  {
    // Constant, but activation cache is mutable
    const network_node& node = network.nodes[i];

    // If this is false, either the neuron is an input or the neuron is not connected
    // to the network. Either way, we do not need to compute anything.
    if (node.steps)
    {
      // We must reset the last activation of the node to 0
      node.activation = 0;

      const uint32_t target_step = progress + node.steps;

      for (; progress < target_step; progress++)
      {
        const network_step step = network.steps[progress];

        node.activation += network.nodes[step.source].activation * step.weight;
      }

      // Once all the propagation steps have been compleated for a node we activate it
      // before compleating the steps for the next node.
      node.activation = activate(node.activation);
    }
  }
}

/**
 * @brief Returns the input activations for the last propagation.
 * 
 * @param network The neural network to retreive input activations for.
 * @return std::vector<float> The input activations
 */
inline std::vector<float> propagation_inputs(const neural_network& network)
{
  std::vector<float> inputs;

  inputs.reserve(network.input_count);

  for (size_t i = 0; i < network.input_count; i++)
  {
    inputs.push_back(network.nodes[network.input_mapping(i)].activation);
  }

  return inputs;
}

/**
 * @brief Returns the output activations for the last propagation.
 * 
 * @param network The neural network to retreive output activations for.
 * @return std::vector<float> The output activations
 */
inline std::vector<float> propagation_outputs(const neural_network& network)
{
  std::vector<float> outputs;

  outputs.reserve(network.output_count);

  for (size_t i = 0; i < network.output_count; i++)
  {
    outputs.push_back(network.nodes[network.output_mapping(i)].activation);
  }

  return outputs;
}


} // namespace enet

#endif