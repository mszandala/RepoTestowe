using SystemTestow.PluginDefinitions;
using SystemTestow.PluginDefinitions.Interfaces;

namespace Plugin.Aquila;

public class Aquila(int population, int iterations, FitnessFunction fitnessFunction, double[][] domain, double alpha, double delta) : IOptimizationAlgorithm
{
    private readonly Random _rng = new();
    private double[][]? _x;
    private double[] y = [];
    private readonly int _dim = domain.Length;
    private int? _currentIteration;
    private int? _fitnessEvaluations;
    
    private List<double> MeanPopulation()
    {
        if (_x is null)
        {
            throw new InvalidOperationException("Algorithm has not been initialized.");
        }
        var mean = new List<double>();
        for (var d = 0; d < _dim; d++)
        {
            var sum = 0.0;
            for (var r = 0; r < population; r++)
                sum += _x[r][d];
            mean.Add(sum / population);
        }
        return mean;
    }
    
    private double[] ApplyBounds(double[] vec)
    {
        var validVector = new List<double>();
        for (var i = 0; i < _dim; i++)
        {
            var v = vec[i];
            if (v < domain[i][0]) v = domain[i][0];
            if (v > domain[i][1]) v = domain[i][1];
            validVector.Add(v);
        }
        return validVector.ToArray();
    }

    
    public void Initialize()
    {
        _x = Enumerable.Range(0, population).Select(_ => Enumerable.Range(0, _dim).Select(i => 
            domain[i][0] + _rng.NextDouble() * (domain[i][1] - domain[i][0])
            ).ToArray()).ToArray();
        _currentIteration = 0;
        _fitnessEvaluations = 0;
    }

    public OptimizationAlgorithmResult Solve()
    {
        if (_x is null || _currentIteration is null || _fitnessEvaluations is null)
        {
            throw new InvalidOperationException("Algorithm has not been initialized.");
        }
        var math = new AquilaInternals(_dim, alpha: alpha, delta: delta);
        
        y = _x.Select(x => fitnessFunction(x)).ToArray();
        _fitnessEvaluations += population;
        var (first, yBest) = _x.Zip(y).MinBy(x => x.Second);
        var xBest = first.ToList();

        for (var t = (int)_currentIteration + 1; t <= iterations; t++)
        {
            for (var i = 0; i < population; i++)
            {
                var xMean = MeanPopulation();
                var xCurrent = _x[i].ToList();
                var yCurrent = y[i];
                double[] xCandidate;
                var rand = _rng.NextDouble();
                if (t <= 2.0 / 3.0 * iterations)
                {
                    xCandidate = rand <= 0.5 ? math.ExpandedExploration(xBest, xMean, t, iterations).ToArray() : math.NarrowedExploration(xBest, _x[_rng.Next(population)].ToList()).ToArray();
                }
                else
                {
                    xCandidate = rand <= 0.5 ? math.ExpandedExploitation(xBest, xMean, domain).ToArray() : math.NarrowedExploitation(xBest, xCurrent, t, iterations).ToArray();
                }
                
                xCandidate = ApplyBounds(xCandidate);
                var yCandidate = fitnessFunction(xCandidate);
                _fitnessEvaluations += 1;

                if (!(yCandidate < yCurrent)) continue;
                _x[i] = xCandidate;
                y[i] = yCandidate;
                if (!(yCandidate < yBest)) continue;
                xBest = xCandidate.ToList();
                yBest = yCandidate;
            }
        }
        
        return new OptimizationAlgorithmResult
        {
            FitnessEvaluations = (int)_fitnessEvaluations,
            XBest = xBest.ToArray(),
            YBest = yBest,
        };
    }
}