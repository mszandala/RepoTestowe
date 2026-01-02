using aquila.Interfaces;

namespace aquila.TestFunctions;

public class Beale : ITestFunction
{
    public string displayName => "Beale";
    public string GetTypstMath()
    {
        return "f(x, y) = (1.5 - x + x y)^2 + (2.25 - x + x y^2)^2 + (2.625 - x + x y ^3)^2";
    }

    public string GetTypstPlot()
    {
        return
            "(x, y) => {calc.pow(1.5 - x + x * y, 2) + calc.pow(2.25 - x + x * y * y, 2) + calc.pow(2.625 - x + x * y * y * y, 2)}";
    }

    public int GetDim()
    {
        return 2;
    }

    public List<double> GetUpperBounds()
    {
        return [4.5, 4.5];
    }

    public List<double> GetLowerBounds()
    {
        return [-4.5, -4.5];
    }

    public double Fitness(List<double> x)
    {
        return double.Pow(1.5 - x[0] + x[0] * x[1], 2) + double.Pow(2.25 - x[0] + x[0] * x[1] * x[1], 2) +
               double.Pow(2.625 - x[0] + x[0] * x[1] * x[1] * x[1], 2);
    }
}