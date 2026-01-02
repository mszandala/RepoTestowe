namespace SystemTestow.PluginDefinitions.Interfaces;

public interface IOptimizationAlgorithm
{
    void Initialize();
    OptimizationAlgorithmResult Solve();
}