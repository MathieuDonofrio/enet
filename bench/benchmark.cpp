#include "benchmark.h"

#include <chromosome.hpp>
#include <neural_network.hpp>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random.hpp>

float activate_ref(float x)
{
  return 1.0f / (1.0 + exp(-4.9f * x));
}

struct GeneData
{
  uint64_t i;
  uint64_t j;
};

void Activate_CMathSigmoid_AsComparaison()
{
  const size_t iterations = 10000000;

  const float samples = 1000;

  BEGIN_BENCHMARK(Activate_CMathSigmoid_AsComparaison);

  for (size_t i = 0; i < iterations / samples; i++)
  {
    for (float x = -2; x < 2; x += 4 / samples)
    {
      benchmark::do_not_optimize(activate_ref(x));
    }
  }

  END_BENCHMARK(iterations, 1);
}

void Activate_LargeRange()
{
  const size_t iterations = 10000000;

  const float samples = 1000;
  const float half_samples = samples / 2;

  BEGIN_BENCHMARK(Activate_LargeRange);

  for (size_t i = 0; i < iterations / samples; i++)
  {
    for (float x = -half_samples; x < half_samples; x++)
    {
      benchmark::do_not_optimize(enet::activate(x));
    }
  }

  END_BENCHMARK(iterations, 1);
}

void Activate_ShortRange()
{
  const size_t iterations = 10000000;

  const float samples = 1000;

  BEGIN_BENCHMARK(Activate_ShortRange);

  for (size_t i = 0; i < iterations / samples; i++)
  {
    for (float x = -2; x < 2; x += 4 / samples)
    {
      benchmark::do_not_optimize(enet::activate(x));
    }
  }

  END_BENCHMARK(iterations, 1);
}

void Random_STDRandInt_AsComparaison()
{
  const size_t iterations = 10000000;

  std::srand(std::time(nullptr));

  BEGIN_BENCHMARK(Random_STDRand_AsComparaison);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(std::rand());
  }

  END_BENCHMARK(iterations, 1);
}

void Random_UInt32()
{
  const size_t iterations = 10000000;

  enet::random r { enet::seed() };

  BEGIN_BENCHMARK(Random_UInt32);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(r.next_uint32());
  }

  END_BENCHMARK(iterations, 1);
}

void Random_UInt32Bounded()
{
  const size_t iterations = 10000000;

  enet::random r { enet::seed() };

  BEGIN_BENCHMARK(Random_UniformBounded);

  for (size_t i = 1; i <= iterations; i++)
  {
    benchmark::do_not_optimize(r.next_uint32(i));
  }

  END_BENCHMARK(iterations, 1);
}

void Random_Float()
{
  const size_t iterations = 10000000;

  enet::random r { enet::seed() };

  BEGIN_BENCHMARK(Random_Float);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(r.next_float());
  }

  END_BENCHMARK(iterations, 1);
}

void Random_FloatBounded()
{
  const size_t iterations = 10000000;

  enet::random r { enet::seed() };

  BEGIN_BENCHMARK(Random_FloatBounded);

  for (size_t i = 1; i <= iterations; i++)
  {
    benchmark::do_not_optimize(r.next_float(i));
  }

  END_BENCHMARK(iterations, 1);
}

void Random_Initalize()
{
  const size_t iterations = 10000000;

  BEGIN_BENCHMARK(Random_Initalize);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(enet::random { i });
  }

  END_BENCHMARK(iterations, 1);
}

void Random_Seed()
{
  const size_t iterations = 10000000;

  BEGIN_BENCHMARK(Random_Seed);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(enet::seed());
  }

  END_BENCHMARK(iterations, 1);
}

void Chromosome_Copy_100()
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c;

  for (size_t i = 0; i < 100; i++)
  {
    c.extend(static_cast<float>(i) * 0.01f, { i, i / 2 });
  }

  const size_t iterations = 10000;

  BEGIN_BENCHMARK(Chromosome_Copy_100);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(chromosome { c });
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(c.size());
}

void Chromosome_Extend()
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c;

  const size_t iterations = 1000000;

  BEGIN_BENCHMARK(Chromosome_Extend);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(c.extend(static_cast<float>(i), { i, i }));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(c.size());
}

void Chromosome_Crossover_100()
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  for (size_t i = 0; i < 100; i++)
  {
    c1.extend(static_cast<float>(i) * 0.01f, { i, i / 2 });
  }

  chromosome c2 { c1 };

  for (size_t i = 100; i < 110; i++)
  {
    c2.extend(static_cast<float>(i) * 0.01f, { i, i / 2 });
  }

  for (auto it = c2.begin(); it != c2.end(); ++it)
  {
    it->value += it->value * 0.5f;
    if (it->innovation % 3 == 0) it->disabled = true;
  }

  const size_t iterations = 10000;

  BEGIN_BENCHMARK(Chromosome_Crossover_100);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(c1.crossover(c2));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(c1.size());
  benchmark::do_not_optimize(c2.size());
}

void Chromosome_Compatibility_10()
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  for (size_t i = 0; i < 10; i++)
  {
    c1.extend((static_cast<float>(i) - 5) * 0.1f, { i, i / 2 });
  }

  chromosome c2 { c1 };

  for (size_t i = 10; i < 12; i++)
  {
    c2.extend((static_cast<float>(i) - 5) * 0.1f, { i, i / 2 });
  }

  for (auto it = c2.begin(); it != c2.end(); ++it)
  {
    it->value += it->value += 0.5f;
  }

  const size_t iterations = 10000;

  BEGIN_BENCHMARK(Chromosome_Compatibility_10);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(c1.compatibility(c2));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(c1.size());
  benchmark::do_not_optimize(c2.size());
}

void Chromosome_Compatibility_100()
{
  using chromosome = enet::chromosome<GeneData>;

  chromosome c1;

  for (size_t i = 0; i < 100; i++)
  {
    c1.extend((static_cast<float>(i) - 5) * 0.01f, { i, i / 2 });
  }

  chromosome c2 { c1 };

  for (size_t i = 100; i < 120; i++)
  {
    c2.extend((static_cast<float>(i) - 5) * 0.01f, { i, i / 2 });
  }

  for (auto it = c2.begin(); it != c2.end(); ++it)
  {
    it->value += it->value += 0.5f;
  }

  const size_t iterations = 10000;

  BEGIN_BENCHMARK(Chromosome_Compatibility_100);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(c1.compatibility(c2));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(c1.size());
  benchmark::do_not_optimize(c2.size());
}

void BrainChromosome_BuildNetwork_3_3_3()
{
  auto chromosome = enet::make_fully_connected({ 3, 3, 3 }, true);

  const size_t iterations = 100000;

  BEGIN_BENCHMARK(BrainChromosome_BuildNeuralNetwork_3_3_3);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(chromosome.build_network());
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
}

void BrainChromosome_BuildNetwork_10_10_10()
{
  auto chromosome = enet::make_fully_connected({ 10, 10, 10 }, true);

  const size_t iterations = 100000;

  BEGIN_BENCHMARK(BrainChromosome_BuildNetwork_10_10_10);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(chromosome.build_network());
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
}

void BrainChromosome_BuildNetwork_20_20_20()
{
  auto chromosome = enet::make_fully_connected({ 20, 20, 20 }, true);

  const size_t iterations = 10000;

  BEGIN_BENCHMARK(BrainChromosome_BuildNetwork_20_20_20);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(chromosome.build_network());
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
}

void BrainChromosome_BuildNetwork_20_20_20_20()
{
  auto chromosome = enet::make_fully_connected({ 20, 20, 20, 20 }, true);

  const size_t iterations = 10000;

  BEGIN_BENCHMARK(BrainChromosome_BuildNetwork_20_20_20_20);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(chromosome.build_network());
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
}

void BrainChromosome_BuildNetwork_100_100_100_100()
{
  auto chromosome = enet::make_fully_connected({ 100, 100, 100, 100 }, true);

  const size_t iterations = 1000;

  BEGIN_BENCHMARK(BrainChromosome_BuildNetwork_100_100_100_100);

  for (size_t i = 0; i < iterations; i++)
  {
    benchmark::do_not_optimize(chromosome.build_network());
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
}

void NeuralNetwork_Propagate_3_3_3()
{
  auto chromosome = enet::make_fully_connected({ 3, 3, 3 }, true);

  enet::neural_network network = chromosome.build_network();

  const size_t iterations = 100000;

  std::vector<float> inputs;

  for (size_t i = 0; i < 3; i++)
  {
    inputs.push_back(enet::tl_rand().next_float());
  }

  BEGIN_BENCHMARK(NeuralNetwork_Propagate_3_3_3);

  for (size_t i = 0; i < iterations; i++)
  {
    enet::propagate(network, inputs);

    benchmark::do_not_optimize(enet::propagation_outputs(network));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
  benchmark::do_not_optimize(network.node_count);
  benchmark::do_not_optimize(network.input_count);
  benchmark::do_not_optimize(network.output_count);
}

void NeuralNetwork_Propagate_10_10_10()
{
  auto chromosome = enet::make_fully_connected({ 10, 10, 10 }, true);

  enet::neural_network network = chromosome.build_network();

  const size_t iterations = 100000;

  std::vector<float> inputs;

  for (size_t i = 0; i < 10; i++)
  {
    inputs.push_back(enet::tl_rand().next_float());
  }

  BEGIN_BENCHMARK(NeuralNetwork_Propagate_10_10_10);

  for (size_t i = 0; i < iterations; i++)
  {
    enet::propagate(network, inputs);

    benchmark::do_not_optimize(enet::propagation_outputs(network));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
  benchmark::do_not_optimize(network.node_count);
  benchmark::do_not_optimize(network.input_count);
  benchmark::do_not_optimize(network.output_count);
}

void NeuralNetwork_Propagate_20_20_20()
{
  auto chromosome = enet::make_fully_connected({ 20, 20, 20 }, true);

  enet::neural_network network = chromosome.build_network();

  const size_t iterations = 10000;

  std::vector<float> inputs;

  for (size_t i = 0; i < 20; i++)
  {
    inputs.push_back(enet::tl_rand().next_float());
  }

  BEGIN_BENCHMARK(NeuralNetwork_Propagate_20_20_20);

  for (size_t i = 0; i < iterations; i++)
  {
    enet::propagate(network, inputs);

    benchmark::do_not_optimize(enet::propagation_outputs(network));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
  benchmark::do_not_optimize(network.node_count);
  benchmark::do_not_optimize(network.input_count);
  benchmark::do_not_optimize(network.output_count);
}

void NeuralNetwork_Propagate_20_20_20_20()
{
  auto chromosome = enet::make_fully_connected({ 20, 20, 20, 20 }, true);

  enet::neural_network network = chromosome.build_network();

  const size_t iterations = 10000;

  std::vector<float> inputs;

  for (size_t i = 0; i < 20; i++)
  {
    inputs.push_back(enet::tl_rand().next_float());
  }

  BEGIN_BENCHMARK(NeuralNetwork_Propagate_20_20_20_20);

  for (size_t i = 0; i < iterations; i++)
  {
    enet::propagate(network, inputs);

    benchmark::do_not_optimize(enet::propagation_outputs(network));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
  benchmark::do_not_optimize(network.node_count);
  benchmark::do_not_optimize(network.input_count);
  benchmark::do_not_optimize(network.output_count);
}

void NeuralNetwork_Propagate_100_100_100_100()
{
  auto chromosome = enet::make_fully_connected({ 100, 100, 100, 100 }, true);

  enet::neural_network network = chromosome.build_network();

  const size_t iterations = 1000;

  std::vector<float> inputs;

  for (size_t i = 0; i < 100; i++)
  {
    inputs.push_back(enet::tl_rand().next_float());
  }

  BEGIN_BENCHMARK(NeuralNetwork_Propagate_100_100_100_100);

  for (size_t i = 0; i < iterations; i++)
  {
    enet::propagate(network, inputs);

    benchmark::do_not_optimize(enet::propagation_outputs(network));
  }

  END_BENCHMARK(iterations, 1);

  benchmark::do_not_optimize(chromosome.size());
  benchmark::do_not_optimize(network.node_count);
  benchmark::do_not_optimize(network.input_count);
  benchmark::do_not_optimize(network.output_count);
}

int main()
{
  Activate_CMathSigmoid_AsComparaison();
  Activate_LargeRange();
  Activate_ShortRange();

  Random_STDRandInt_AsComparaison();
  Random_UInt32();
  Random_UInt32Bounded();
  Random_Float();
  Random_FloatBounded();
  Random_Initalize();
  Random_Seed();

  Chromosome_Copy_100();
  Chromosome_Extend();
  Chromosome_Crossover_100();
  Chromosome_Compatibility_10();
  Chromosome_Compatibility_100();

  BrainChromosome_BuildNetwork_3_3_3();
  BrainChromosome_BuildNetwork_10_10_10();
  BrainChromosome_BuildNetwork_20_20_20();
  BrainChromosome_BuildNetwork_20_20_20_20();
  BrainChromosome_BuildNetwork_100_100_100_100();

  NeuralNetwork_Propagate_3_3_3();
  NeuralNetwork_Propagate_10_10_10();
  NeuralNetwork_Propagate_20_20_20();
  NeuralNetwork_Propagate_20_20_20_20();
  NeuralNetwork_Propagate_100_100_100_100();

  return 0;
}