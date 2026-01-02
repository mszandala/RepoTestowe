using SystemTestow.PluginDefinitions;
using SystemTestow.PluginDefinitions.Interfaces;

namespace Plugin.Aquila;

public class Plugin : BasePlugin
{
    public override IEnumerable<ITestFunction> TestFunctions { get; } = [];
    public override IEnumerable<IOptimizationAlgorithmInfo> OptimizationAlgorithms { get; } = [
        new AquilaInfo(),
    ];
}