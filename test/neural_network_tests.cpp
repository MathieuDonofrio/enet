#include <chromosome.hpp>
#include <gtest/gtest.h>
#include <neural_network.hpp>
#include <cmath>

float activate_ref(float x)
{
  return 1 / (1 + exp(-4.9f * x));
}

TEST(NeuralNetwork, Activate_HighValues_MaxError)
{
    for(float x = -1000000; x < 1000000; x++)
    {
        ASSERT_NEAR(enet::activate(x), activate_ref(x), MAX_ACTIVATION_ERROR);
    }
}

TEST(NeuralNetwork, Activate_LowValues_MaxError)
{
    for(float x = -2; x < 2; x += 0.0001f)
    {
        ASSERT_NEAR(enet::activate(x), activate_ref(x), MAX_ACTIVATION_ERROR);
    }
}

TEST(NeuralNetwork, Activate_LowValues_AverageError)
{
    float error_sum = 0;
    int iterations = 0;

    for(float x = -2; x < 2; x += 0.0001f)
    {
        error_sum += abs(enet::activate(x) - activate_ref(x));
        iterations++;
    }

    const float average_error = error_sum / iterations;

    ASSERT_LE(average_error, AVERAGE_ACTIVATION_ERROR);
}

TEST(NeuralNetwork, Propagate_SingleSynapse)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(1, enet::neuron_type::OUTPUT);

  c.add_synapse(1.0f, 0, 1);

  enet::neural_network network = c.build_network();

  std::vector<float> inputs = { 1.0f };

  enet::propagate(network, inputs);

  std::vector<float> outputs = enet::propagation_outputs(network);

  ASSERT_EQ(outputs[0], enet::activate(1.0f));
}

TEST(NeuralNetwork, Propagate_WithHidden)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(0.5f, enet::neuron_type::HIDDEN);
  c.add_neuron(1, enet::neuron_type::OUTPUT);

  c.add_synapse(0.5f, 0, 1);
  c.add_synapse(1.5f, 1, 2);

  enet::neural_network network = c.build_network();

  std::vector<float> inputs = { 1.5f };

  enet::propagate(network, inputs);

  std::vector<float> outputs = enet::propagation_outputs(network);

  float expected = enet::activate(enet::activate(1.5f * 0.5f) * 1.5f);

  ASSERT_EQ(outputs[0], expected);
}

TEST(NeuralNetwork, Propagate_DoubleInput)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(1, enet::neuron_type::OUTPUT);

  c.add_synapse(0.5f, 0, 2);
  c.add_synapse(1.5f, 0, 2);

  enet::neural_network network = c.build_network();

  std::vector<float> inputs = { 2.0f, 1.5f };

  enet::propagate(network, inputs);

  std::vector<float> outputs = enet::propagation_outputs(network);

  float expected = enet::activate(0.5f * 2.0f + 1.5f * 1.5f);

  ASSERT_EQ(outputs[0], expected);
}

TEST(NeuralNetwork, Propagate_DoubleInputMultipleSynapase)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(1, enet::neuron_type::OUTPUT);

  c.add_synapse(0.5f, 0, 2);
  c.add_synapse(1.2f, 1, 2);
  c.add_synapse(1.5f, 0, 2);

  enet::neural_network network = c.build_network();

  std::vector<float> inputs = { 2.0f, 1.5f };

  enet::propagate(network, inputs);

  std::vector<float> outputs = enet::propagation_outputs(network);

  float expected = enet::activate(0.5f * 2.0f + 1.2f * 2.0f + 1.5f * 1.5f);

  ASSERT_EQ(outputs[0], expected);
}

TEST(NeuralNetwork, Propagate_WithHidenAndSkip)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(0, enet::neuron_type::HIDDEN);
  c.add_neuron(1, enet::neuron_type::OUTPUT);

  c.add_synapse(0.5f, 0, 2);
  c.add_synapse(1.2f, 1, 2);
  c.add_synapse(1.5f, 0, 1);

  enet::neural_network network = c.build_network();

  std::vector<float> inputs = { 1.2f };

  enet::propagate(network, inputs);

  std::vector<float> outputs = enet::propagation_outputs(network);

  float expected = enet::activate(enet::activate(1.5f * 1.2f) + 0.5f * 1.2f);

  ASSERT_EQ(outputs[0], expected);
}

TEST(NeuralNetwork, Propagate_SingleWeightFullyConnected)
{
  using chromosome = enet::brain_chromosome;

  chromosome c = enet::make_fully_connected({3, 3, 3});

  enet::neural_network network = c.build_network();

  std::vector<float> inputs = { 1.0f, 1.0f, 1.0f };

  enet::propagate(network, inputs);

  std::vector<float> outputs = enet::propagation_outputs(network);

  ASSERT_EQ(outputs[0], 1);
  ASSERT_EQ(outputs[1], 1);
  ASSERT_EQ(outputs[2], 1);
}