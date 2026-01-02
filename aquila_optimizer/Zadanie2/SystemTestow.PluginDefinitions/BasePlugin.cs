using SystemTestow.PluginDefinitions.Interfaces;

namespace SystemTestow.PluginDefinitions;

public abstract class BasePlugin
{
    public abstract IEnumerable<ITestFunction> TestFunctions { get; }
    public abstract IEnumerable<IOptimizationAlgorithmInfo> OptimizationAlgorithms { get; }
}