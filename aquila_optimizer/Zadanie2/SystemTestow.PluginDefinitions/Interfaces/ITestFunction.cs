namespace SystemTestow.PluginDefinitions.Interfaces;

public interface ITestFunction
{
    public string Name { get; }
    public string TypstMath { get; }
    public int Dim { get; }
    public FitnessFunction Fitness { get;  }
}