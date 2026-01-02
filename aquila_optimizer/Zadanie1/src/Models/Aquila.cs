using aquila.Models;

namespace Models.Aquila;

using aquila;
using Matrix = List<List<double>>;

public class Aquila
{
    private readonly int _nrow;
    private readonly int _dim;
    private readonly int _maxIterations;
    private readonly double _alpha;
    private readonly double _delta;
    private readonly List<double> _upperBounds;
    private readonly List<double> _lowerBounds;
    private readonly Func<List<double>, double> _fitnessFunction;
    private Matrix _X;

    public Aquila(
            int nrow,
            int dim,
            double alpha,
            double delta,
            List<double> upperBounds,
            List<double> lowerBounds,
            Func<List<double>, double> fitnessFunction,
            int maxIterations = 100
            )
    {
        _nrow = nrow;
        _dim = dim;
        _alpha = alpha;
        _delta = delta;
        _maxIterations = maxIterations;
        _fitnessFunction = fitnessFunction;
        _upperBounds = upperBounds;
        _lowerBounds = lowerBounds;

        _X = PopulationInit();
    }

    private Matrix PopulationInit()
    {
        var rnd = new Random();
        Matrix X = Enumerable.Range(0, _nrow)
            .Select(_ => Enumerable.Range(0, _dim)
                    .Select(idx =>
                        {
                            return _lowerBounds[idx] + rnd.NextDouble() * (_upperBounds[idx] - _lowerBounds[idx]);
                        }
                        ).ToList()
                   ).ToList();
        return X;
    }

    private List<double> PopulationBest()
    {
        double bestFitValue = _fitnessFunction(_X[0]);
        List<double> X_best = _X[0];

        // Dodac jakies sprawdzanie danych?
        if (_X.Count == 1)
        {
            return X_best;
        }

        for (int row = 1; row < _nrow; ++row)
        {
            double candidateValue = _fitnessFunction(_X[row]);
            // Chyba bedzie zalezec od zadania
            // Ustawione na min funkcji.
            if (bestFitValue > candidateValue)
            {
                X_best = _X[row];
            }
        }
        return X_best;
    }


    private List<double> MeanPopulation()
    {
        List<double> mean = new List<double>();
        for (int d = 0; d < _dim; d++)
        {
            double sum = 0.0;
            for (int r = 0; r < _nrow; r++)
                sum += _X[r][d];
            mean.Add(sum / _nrow);
        }
        return mean;
    }

    private List<double> ApplyBounds(List<double> vec)
    {
        List<double> validVector = new List<double>();
        for (int i = 0; i < _dim; i++)
        {
            double v = vec[i];
            if (v < _lowerBounds[i]) v = _lowerBounds[i];
            if (v > _upperBounds[i]) v = _upperBounds[i];
            validVector.Add(v);
        }
        return validVector;
    }

    public (List<double> Xbest, double XbestFitness) Predict()
    {
        int T = _maxIterations;
        var rnd = new Random();

        var math = new AquilaMath(_dim, alpha: _alpha, delta: _delta); // jeśli przeniesiemy AquilaMath to trzeba będzie zmienić to i Step 1-4

        List<double> Xbest = PopulationBest();
        double XbestFitness = _fitnessFunction(Xbest);

        for (int t = 1; t <= T; t++)
        {
            List<double> XM = MeanPopulation();

            for (int i = 0; i < _nrow; i++)
            {
                List<double> Xcurrent = new List<double>();
                for (int j = 0; j < _dim; j++)
                {
                    Xcurrent.Add(_X[i][j]);
                }

                double currentFitness = _fitnessFunction(Xcurrent);
                double rand = rnd.NextDouble();
                List<double>? candidate = null;

                if (t <= 2.0 / 3.0 * T)
                {
                    if (rand <= 0.5)
                    {
                        // Step 1
                        candidate = math.ExpandedExploration(Xbest, XM, t, _maxIterations);
                    }
                    else
                    {
                        // Step 2
                        candidate = math.NarrowedExploration(Xbest, _X[rnd.Next(_nrow)]);
                    }
                }
                else
                {
                    if (rand <= 0.5)
                    {
                        // Step 3
                        candidate = math.ExpandedExploitation(Xbest, XM, _upperBounds, _lowerBounds);
                    }
                    else
                    {
                        // Step 4
                        candidate = math.NarrowedExploitation(Xbest, Xcurrent, t, _maxIterations);
                    }
                }

                if (candidate != null)
                {
                    candidate = ApplyBounds(candidate);
                    double candidateFitness = _fitnessFunction(candidate);


                    if (candidateFitness < currentFitness)
                    {
                        _X[i] = candidate;

                        if (candidateFitness < XbestFitness)
                        {
                            for (int j = 0; j < _dim; j++)
                            {
                                Xbest[j] = candidate[j];
                            }

                            XbestFitness = candidateFitness;
                        }
                    }
                }
                // Logging.LogX(t, Xcurrent);
            }
            // tu może można jakiś zapis do raportu dla t-tej iteracji
        }
        return (Xbest, XbestFitness); // zwraca tylko Xbest, bo czy będziemy potrzebować też gdzieś XbestFitness? 
    }
}
