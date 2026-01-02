using aquila.Interfaces;

namespace aquila.TestFunctions;

public class Sphere(int dim) : ITestFunction
{
    
    public string displayName => "Sphere";
    public string GetTypstMath()
    {
        return "f(x, y) = x^2 + y^2";
    }

    public string GetTypstPlot()
    {
        return "(x, y) => {x * x + y * y}";
    }

    public int GetDim()
    {
        return dim;
    }

    public List<double> GetUpperBounds()
    {
        // Na wikipedii jest -inf do +inf
        return Enumerable.Repeat(50000.0, dim).ToList();
    }

    public List<double> GetLowerBounds()
    {
        return Enumerable.Repeat(-50000.0, dim).ToList();
    }

    public double Fitness(List<double> x)
    {
        return x.Select(xi => Math.Pow(xi, 2)).Sum();
    }
}