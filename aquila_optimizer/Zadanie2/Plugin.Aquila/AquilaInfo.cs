using SystemTestow.PluginDefinitions;
using SystemTestow.PluginDefinitions.Interfaces;

namespace Plugin.Aquila;

public class AquilaInfo : IOptimizationAlgorithmInfo
{
    public string Name { get; } = "Aquila Optimizer";
    public ParamsInfo[] ParamsInfo { get; } = [
        new()
        {
            Name = "alpha",
            Description = "alpha",
            TypstMath = "alpha",
            LowerBoundary = 0.1,
            UpperBoundary = 0.9
        },
        new()
        {
            Name = "delta",
            Description = "delta",
            TypstMath = "delta",
            LowerBoundary = 0.1,
            UpperBoundary = 0.9
        },
    ];
    public IOptimizationAlgorithm Create(int population, int iterations, FitnessFunction fitnessFunction,
        double[][] domain, params double[] parameters)
    {
        return new Aquila( population, iterations, fitnessFunction, domain, parameters[0], parameters[1]);
    }
}