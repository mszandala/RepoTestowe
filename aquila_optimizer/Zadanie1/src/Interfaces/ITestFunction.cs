namespace aquila.Interfaces;

public interface ITestFunction
{
    string displayName { get; }
    public string GetTypstMath();
    public string GetTypstPlot();
    public int GetDim();
    public List<double> GetUpperBounds();
    public List<double> GetLowerBounds();
    public double Fitness(List<double> x);
}