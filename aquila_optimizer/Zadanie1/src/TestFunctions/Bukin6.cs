using aquila.Interfaces;

namespace aquila.TestFunctions;

public class Bukin6 : ITestFunction
{
    public string displayName => "Bukin6";

    public string GetTypstMath()
    {
        return "f(x, y) = 100 sqrt(abs(y-0.01 x^2)) + 0.01 abs(x + 10)";
    }

    public string GetTypstPlot()
    {
        return "(x, y) => {100 * calc.sqrt(calc.abs(y - 0.01 * x * x)) + 0.01 * calc.abs(x + 10)}";
    }

    public int GetDim()
    {
        return 2;
    }

    public List<double> GetUpperBounds()
    {
        return [-5, 3];
    }

    public List<double> GetLowerBounds()
    {
        return [-15, -3];
    }

    public double Fitness(List<double> x)
    {
        return 100 * double.Sqrt(double.Abs(x[1] - 0.01 * x[0] * x[0])) + 0.01 * double.Abs(x[0] + 10);
    }
}