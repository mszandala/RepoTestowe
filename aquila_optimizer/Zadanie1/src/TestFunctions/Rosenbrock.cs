using aquila.Interfaces;

namespace aquila.TestFunctions;

public class Rosenbrock(int dim) : ITestFunction
{
    public string displayName => "Rosenbrock";
    public string GetTypstMath()
    {
        return "f(x, y) = 100(y - x^2) + (1 - x)^2";
    }

    public string GetTypstPlot()
    {
        return "(x, y) => {100 * (y - x * x) + (1 - x) * (1 - x)}";
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
        double sum = 0.0;
        for (int i = 0; i < dim - 1; i++)
        {
            double xi = x[i];
            double xi1 = x[i + 1];
            sum += Math.Pow(1 - xi, 2) + 100.0 * Math.Pow(xi1 - xi * xi, 2);
        }

        return sum;
    }
}