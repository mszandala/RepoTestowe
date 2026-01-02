namespace SystemTestow.PluginDefinitions.Interfaces;

public interface IOptimizationAlgorithmInfo
{
    public string Name { get; }
    ParamsInfo[] ParamsInfo { get; }
    IOptimizationAlgorithm Create(int population, int iterations, FitnessFunction fitnessFunction, double[][] domain, params double[] parameters);
}