using aquila.Interfaces;

namespace aquila.TestFunctions;

public class Rastrigin(int dim) : ITestFunction
{
    public string displayName => "Rastrigin";
    public string GetTypstMath()
    {
        return "f(x, y) = 20 + x^2 - 10 cos(2 pi x) + y^2 - 10 cos(2 pi y)";
    }

    public string GetTypstPlot()
    {
        return "(x, y) => {20 + x * x + y * y - 10 * (calc.cos(2 * calc.pi * x) + calc.cos(2 * calc.pi * y))}";
    }

    public int GetDim()
    {
        return dim;
    }

    public List<double> GetUpperBounds()
    {
        return Enumerable.Repeat(5.12, dim).ToList();
    }

    public List<double> GetLowerBounds()
    {
        return Enumerable.Repeat(-5.12, dim).ToList();
    }

    public double Fitness(List<double> x)
    {
        const double a = 10.0;
        return a * GetDim() + x.Select(xi => xi * xi - a * double.Cos(2 * double.Pi * xi)).Sum();
    }
}