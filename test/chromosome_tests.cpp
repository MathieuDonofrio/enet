
#include <chromosome.hpp>
#include <gtest/gtest.h>

struct GeneData
{
  double x;
};

TEST(BasicChromosome, Empty_OnInitialization_True)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c;

  ASSERT_TRUE(c.empty());
  ASSERT_EQ(c.size(), 0);
}

TEST(BasicChromosome, Extend_Single_SizeIncrease)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c;

  c.extend(0.5f, {});

  ASSERT_FALSE(c.empty());
  ASSERT_EQ(c.size(), 1);
}

TEST(BasicChromosome, Extend_Double_SizeIncrease)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c;

  c.extend(0.5f, {});
  c.extend(0.5f, {});

  ASSERT_FALSE(c.empty());
  ASSERT_EQ(c.size(), 2);
}

TEST(BasicChromosome, Extend_Many_SizeIncrease)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c;

  float amount = 100;

  for (float i = 0; i < amount; i++)
  {
    c.extend(i / amount, {});
  }

  ASSERT_FALSE(c.empty());
  ASSERT_EQ(c.size(), amount);
}

TEST(BasicChromosome, Find_Single_Equal)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c;

  auto i = c.extend(0.5f, {});

  ASSERT_EQ(c.find(i)->value, 0.5f);
}

TEST(BasicChromosome, Find_Many_Equal)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c;

  float amount = 100;

  for (float i = 0; i < amount; i++)
  {
    c.extend(i / amount, {});
  }

  for (auto it = c.begin(); it != c.end(); ++it)
  {
    ASSERT_EQ(it->value, c.find(it->innovation)->value);
  }
}

TEST(BasicChromosome, CopyConstructor_SingleGene_SameSize)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  c1.extend(0.5f, {});

  chromosome c2 = { c1 };

  ASSERT_EQ(c2.size(), c1.size());
}

TEST(BasicChromosome, CopyConstructor_SingleGene_SameValue)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  c1.extend(0.5f, {});

  chromosome c2 = { c1 };

  ASSERT_EQ(c2.begin()->value, 0.5f);
  ASSERT_EQ(c2.begin()->innovation, c1.begin()->innovation);
  ASSERT_EQ(c2.begin()->disabled, c1.begin()->disabled);
}

TEST(BasicChromosome, Crossover_SameSingleGene_Same)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  c1.extend(0.5f, {});

  chromosome c2 = { c1 };

  chromosome c3 = c1.crossover(c2);

  ASSERT_EQ(c3.size(), c1.size());
  ASSERT_EQ(c3.begin()->value, 0.5f);
  ASSERT_EQ(c3.begin()->value, c1.begin()->value);
  ASSERT_EQ(c3.begin()->innovation, c1.begin()->innovation);
  ASSERT_EQ(c2.begin()->disabled, c1.begin()->disabled);
}

TEST(BasicChromosome, Crossover_DifferentSingleGene_Same)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  c1.extend(0.5f, {});

  chromosome c2 = { c1 };

  chromosome c3 = c1.crossover(c2);

  ASSERT_EQ(c3.size(), c1.size());
  ASSERT_EQ(c3.begin()->value, 0.5f);
  ASSERT_EQ(c3.begin()->value, c1.begin()->value);
  ASSERT_EQ(c3.begin()->innovation, c1.begin()->innovation);
  ASSERT_EQ(c2.begin()->disabled, c1.begin()->disabled);
}

TEST(BasicChromosome, Crossover_DifferentSingleGene_CorrectMutation)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  c1.extend(0.5f, {});

  chromosome c2 = { c1 };

  c2.begin()->value = 1.0f;
  c2.begin()->disabled = true;

  enet::random predictor = enet::tl_rand();

  chromosome c3 = c1.crossover(c2);

  uint32_t r = predictor.next_uint32();

  float predict_value = enet::extract_bool(r, 0) ? 1.0f : 0.5f;
  bool predict_disabled = enet::extract_bool(r, 2);

  ASSERT_EQ(c3.size(), c1.size());
  ASSERT_EQ(c3.begin()->value, predict_value);
  ASSERT_EQ(c3.begin()->disabled, predict_disabled);
  ASSERT_EQ(c3.begin()->innovation, c1.begin()->innovation);
}

TEST(BasicChromosome, Crossover_DifferentInnovations_NothingToCrossover)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  c1.extend(0.5f, {});

  chromosome c2;

  c2.extend(1.0f, {});

  chromosome c3 = c1.crossover(c2);

  ASSERT_EQ(c3.size(), 1);
  ASSERT_EQ(c3.begin()->value, c1.begin()->value);
}

TEST(BasicChromosome, Crossover_MultipleInnovations)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  for (float i = 0; i < 100; i++)
  {
    c1.extend(i, {});
  }

  chromosome c2 { c1 };

  enet::random predictor = enet::tl_rand();

  std::vector<float> predict_value;
  std::vector<float> predict_disabled;

  for (auto it = c2.begin(); it != c2.end(); ++it)
  {
    uint32_t r = predictor.next_uint32();

    predict_value.push_back(enet::extract_bool(r, 0) ? it->value + 0.5f : it->value);
    predict_disabled.push_back((it->innovation % 2 == 0) ? enet::extract_bool(r, 2) : false);

    it->value += 0.5f;
    if (it->innovation % 2 == 0) it->disabled = true;
  }

  c2.extend(-1.0f, {});

  chromosome c3 = c1.crossover(c2);

  ASSERT_EQ(c3.size(), 100);
  ASSERT_EQ(c3.size(), c1.size());

  for (size_t i = 0; i < c1.size(); i++)
  {
    auto it = c3.begin() + i;

    ASSERT_EQ(it->value, predict_value[i]);
    ASSERT_EQ(it->disabled, predict_disabled[i]);
  }
}

TEST(BasicChromosome, Compatibility_NoGenes_HighestCompatibility)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  chromosome c2;

  float compatibility = c1.compatibility(c2);

  ASSERT_EQ(compatibility, ENET_HIGHEST_COMPATIBILITY);
}

TEST(BasicChromosome, Compatibility_SameSingleGene_HighestCompatibility)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  c1.extend(0.5f, {});

  chromosome c2 { c1 };

  float compatibility = c1.compatibility(c2);

  ASSERT_EQ(compatibility, ENET_HIGHEST_COMPATIBILITY);
}

TEST(BasicChromosome, Compatibility_SameMultipleGenes_HighestCompatibility)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  for (float i = 1; i <= 100; i++)
  {
    c1.extend(1.0f / i, {});
  }

  chromosome c2 { c1 };

  float compatibility = c1.compatibility(c2);

  ASSERT_EQ(compatibility, ENET_HIGHEST_COMPATIBILITY);
}

TEST(BasicChromosome, Compatibility_AllExcess_LowestCompatibility)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  for (float i = 1; i <= 10; i++)
  {
    c1.extend(1.0f / i, {});
  }

  chromosome c2;

  float compatibility = c1.compatibility(c2);

  ASSERT_EQ(compatibility, ENET_LOWEST_COMPATIBILITY);
}

TEST(BasicChromosome, Compatibility_AllExcessOrDisjoint_LowestCompatibility)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  for (float i = 1; i <= 10; i++)
  {
    c1.extend(1.0f / i, {});
  }

  chromosome c2;

  for (float i = 1; i <= 5; i++)
  {
    c2.extend(1.0f / i, {});
  }

  float compatibility = c1.compatibility(c2);

  ASSERT_EQ(compatibility, ENET_LOWEST_COMPATIBILITY);
}

TEST(BasicChromosome, Compatibility_SameGenesDifferentValues)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  for (float i = 1; i <= 100; i++)
  {
    c1.extend(1.0f / i, {});
  }

  chromosome c2 { c1 };

  for (auto it = c2.begin(); it != c2.end(); ++it)
  {
    it->value += 1;
  }

  float compatibility = c1.compatibility(c2);

  ASSERT_EQ(compatibility, 0.5f);
}

TEST(BasicChromosome, Compatibility_Complex)
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  c1.extend(-1.0f, {});
  c1.extend(-0.5f, {});
  c1.extend(0, {});
  c1.extend(0.5f, {});
  c1.extend(1.0f, {});

  chromosome c2 { c1 };

  c2.extend(1.0f, {});
  c2.extend(0.0f, {});

  for (auto it = c2.begin(); it != c2.end(); ++it)
  {
    it->value += 0.5f;
  }

  float expected = (5.0f / 7.0f) * (1 - (0.5f / ENET_MAX_COMPATIBILITY_VALUE_DIFF));

  float compatibility = c1.compatibility(c2);

  ASSERT_EQ(compatibility, expected);
}

TEST(BrainChromosome, NeuronCount_OnInitialization_Empty)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  ASSERT_EQ(0, c.neuron_count());
  ASSERT_EQ(0, c.sensors());
  ASSERT_EQ(0, c.outputs());
}

TEST(BrainChromosome, AddNeuron_SingleHidden_SizeIncrease)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::HIDDEN);

  ASSERT_EQ(1, c.neuron_count());
  ASSERT_EQ(0, c.sensors());
  ASSERT_EQ(0, c.outputs());
}

TEST(BrainChromosome, AddNeuron_SingleSensor_SizeIncrease)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::SENSOR);

  ASSERT_EQ(1, c.neuron_count());
  ASSERT_EQ(1, c.sensors());
  ASSERT_EQ(0, c.outputs());
}

TEST(BrainChromosome, AddNeuron_SingleOutput_SizeIncrease)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::OUTPUT);

  ASSERT_EQ(1, c.neuron_count());
  ASSERT_EQ(0, c.sensors());
  ASSERT_EQ(1, c.outputs());
}

TEST(BrainChromosome, AddNeuron_MultipleHidden_SizeIncrease)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  size_t amount = 10000;

  for (size_t i = 0; i < amount; i++)
  {
    c.add_neuron(0, enet::neuron_type::HIDDEN);
  }

  ASSERT_EQ(amount, c.neuron_count());
  ASSERT_EQ(0, c.sensors());
  ASSERT_EQ(0, c.outputs());
}

TEST(BrainChromosome, AddNeuron_OfEveryType_SizeIncrease)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::HIDDEN);
  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(0, enet::neuron_type::HIDDEN);
  c.add_neuron(0, enet::neuron_type::OUTPUT);
  c.add_neuron(0, enet::neuron_type::HIDDEN);
  c.add_neuron(0, enet::neuron_type::OUTPUT);
  c.add_neuron(0, enet::neuron_type::OUTPUT);
  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(0, enet::neuron_type::HIDDEN);
  c.add_neuron(0, enet::neuron_type::SENSOR);

  ASSERT_EQ(11, c.neuron_count());
  ASSERT_EQ(4, c.sensors());
  ASSERT_EQ(3, c.outputs());
}

TEST(BrainChromosome, AddSynapse_Single_SizeIncrease)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(1, enet::neuron_type::OUTPUT);

  ASSERT_EQ(c.get_neuron(0).outgoing.size(), 0);
  ASSERT_EQ(c.get_neuron(0).incomming.size(), 0);
  ASSERT_EQ(c.get_neuron(1).outgoing.size(), 0);
  ASSERT_EQ(c.get_neuron(1).incomming.size(), 0);

  c.add_synapse(1.0f, 0, 1);

  ASSERT_EQ(1, c.size());
  ASSERT_EQ(c.get_neuron(0).outgoing.size(), 1);
  ASSERT_EQ(c.get_neuron(0).incomming.size(), 0);
  ASSERT_EQ(c.get_neuron(1).outgoing.size(), 0);
  ASSERT_EQ(c.get_neuron(1).incomming.size(), 1);
}

TEST(BrainChromosome, AddSynapse_Double_SizeIncrease)
{
  using chromosome = enet::brain_chromosome;

  chromosome c;

  c.add_neuron(0, enet::neuron_type::SENSOR);
  c.add_neuron(1, enet::neuron_type::OUTPUT);

  c.add_synapse(1.0f, 0, 1);
  c.add_synapse(0.5f, 0, 1);

  ASSERT_EQ(2, c.size());
  ASSERT_EQ(c.get_neuron(0).outgoing.size(), 2);
  ASSERT_EQ(c.get_neuron(0).incomming.size(), 0);
  ASSERT_EQ(c.get_neuron(1).outgoing.size(), 0);
  ASSERT_EQ(c.get_neuron(1).incomming.size(), 2);
}

TEST(BrainChromosome, MakeFullyConnected_3_3)
{
  using chromosome = enet::brain_chromosome;

  chromosome c = enet::make_fully_connected({ 3, 3 });

  ASSERT_EQ(3 * 3, c.size());
  ASSERT_EQ(3 + 3, c.neuron_count());
}

TEST(BrainChromosome, MakeFullyConnected_3_3_3)
{
  using chromosome = enet::brain_chromosome;

  chromosome c = enet::make_fully_connected({ 3, 3, 3 });

  ASSERT_EQ(3 * 3 + 3 * 3, c.size());
  ASSERT_EQ(3 + 3 + 3, c.neuron_count());
}

TEST(BrainChromosome, MakeFullyConnected_10_10_10)
{
  using chromosome = enet::brain_chromosome;

  chromosome c = enet::make_fully_connected({ 10, 10, 10 });

  ASSERT_EQ(10 * 10 + 10 * 10, c.size());
  ASSERT_EQ(10 + 10 + 10, c.neuron_count());
}

TEST(BrainChromosome, BuildNetwork_3_3)
{
  using chromosome = enet::brain_chromosome;

  chromosome c = enet::make_fully_connected({ 3, 3 });

  enet::neural_network network = c.build_network();

  ASSERT_EQ(3 * 3, network.step_count());
  ASSERT_EQ(3 + 3, network.node_count);
}

TEST(BrainChromosome, BuildNetwork_3_3_3)
{
  using chromosome = enet::brain_chromosome;

  chromosome c = enet::make_fully_connected({ 3, 3, 3 });

  enet::neural_network network = c.build_network();

  ASSERT_EQ(3 * 3 + 3 * 3, network.step_count());
  ASSERT_EQ(3 + 3 + 3, network.node_count);
}

TEST(BrainChromosome, BuildNetwork_10_10_10)
{
  using chromosome = enet::brain_chromosome;

  chromosome c = enet::make_fully_connected({ 10, 10, 10 });

  enet::neural_network network = c.build_network();

  ASSERT_EQ(10 * 10 + 10 * 10, network.step_count());
  ASSERT_EQ(10 + 10 + 10, network.node_count);
}