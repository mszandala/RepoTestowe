using MathNet.Numerics;
using MathNet.Numerics.Distributions;

namespace aquila.Models;

// To jest pewnie do przeniesienia jak się pojawi faktyczna klasa z optymalizatora
public class AquilaMath(int dim, double s = 0.01, double alpha = 0.1, double beta = 1.5, double delta = 0.1, double bigU = 0.00565, double omega = 0.005, double r1 = 10)
{
    private readonly Random _random = new();
    private readonly Normal _normal = new(0, 1);

    private readonly double _sigma = SpecialFunctions.Gamma(1.0 + beta) * double.Sin(double.Pi * beta / 2.0) / (SpecialFunctions.Gamma((1.0 + beta)/2.0) * beta * double.Pow(2.0, (beta - 1.0)/2.0));
    private readonly List<double> _d1 = Enumerable.Range(1, dim).Select(x => (double)x).ToList(); 
    
    public List<double> ExpandedExploration(List<double> xBest, List<double> xMean, double t, double T)
    {
        var a1 = 1.0 - t / T;
        var a2 = _random.NextDouble();
        return xBest.Zip(xMean)
            .Select(((double best, double mean) x) => x.best * a1 + (x.mean - x.best) * a2)
            .ToList();
    }

    private double Levy()
    {
        var u = _normal.Sample();
        var v = _normal.Sample();
        return s * u * _sigma / double.Pow(double.Abs(v), 1.0/beta);
    }
    
    public List<double> NarrowedExploration(List<double> xBest, List<double> xRandom)
    {
        var a1 = Levy();
        var a2 = _random.NextDouble();
        return xBest.Zip(xRandom).Zip(_d1, (x, y) => (x.First, x.Second, y))
            .Select(((double best, double random, double d1) x) =>
            {
                var r = r1 + bigU * x.d1;
                var theta = -omega * x.d1 + 3.0 * double.Pi / 2.0;
                var spiralY = r * double.Cos(theta);
                var spiralX = r * double.Sin(theta);
                return x.best * a1 + x.random + (spiralY - spiralX) * a2;
            }).ToList();
    }

    public List<double> ExpandedExploitation(List<double> xBest, List<double> xMean, List<double> upperBounds,
        List<double> lowerBounds)
    {
        var a1 = _random.NextDouble();
        var a2 = _random.NextDouble();
        return xBest.Zip(xMean).Zip(upperBounds.Zip(lowerBounds), (x, y) => (x.First, x.Second, y.First, y.Second)).Select((
            (double best, double mean, double upper, double lower) x) => (x.best - x.mean) * alpha - a1 + ((x.upper - x.lower) * a2 + x.lower) * delta
        ).ToList();
    }

    public List<double> NarrowedExploitation(List<double> xBest, List<double> xPrev, double t, double T)
    {
        var qf = double.Pow(t, (2.0 * _random.NextDouble() - 1.0) / double.Pow(1.0 - T, 2.0));
        var g1 = 2.0 * _random.NextDouble() - 1.0;
        var g2 = 2.0 * (1.0 - t / T);
        var a1 = _random.NextDouble();
        var a2 = Levy();
        var a3 = _random.NextDouble();
        return xBest.Zip(xPrev).Select(((double best, double prev) x) => qf * x.best - g1 * x.prev * a1 - g2 * a2 + a3 * g1).ToList();
    }
}