using System.Text;
using aquila.Interfaces;
using aquila.TestFunctions;
using MathNet.Numerics.Statistics;
using Models.Aquila;

namespace aquila;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("=== Aquila Optimizer ===");

        const int n = 100;

        List<ITestFunction> testFunctions =
        [
            new Beale(),
            new Bukin6(),
            new Rastrigin(2),
            new Rastrigin(5),
            new Rastrigin(10),
            new Rosenbrock(2),
            new Rosenbrock(5),
            new Rosenbrock(10),
            new Sphere(2),
            new Sphere(5),
            new Sphere(10),
            new Sphere(20)
        ];

        List<int> populations = [10, 20, 40, 80];
        List<int> iterationCounts = [5, 10, 20, 40, 60, 80];
        List<double> alphas = [0.1, 0.5, 0.9];
        List<double> deltas = [0.1, 0.5, 0.9];

        Logging.LogText("Algorytm,Funkcja Testowa,Liczba szukanych parametrów,alpha,delta,Liczba iteracji,Rozmiar populacji,Znalezione minimum,Odchylenie standardowe poszukiwanych parametrów,Średnie wartości poszukiwanych parametrów,Wartość funkcji celu,Odchylenie standardowe wartości funkcji celu,Średnia wartość funkcji celu,Najgorsza wartość funkcji celu");

        foreach (var testFunction in testFunctions)
        {
            foreach (var population in populations)
            {
                foreach (var iterations in iterationCounts)
                {
                    foreach (var alpha in alphas)
                    {
                        foreach (var delta in deltas)
                        {
                            var progressUpdate = new StringBuilder(testFunction.displayName);
                            progressUpdate.Append(" dim ");
                            progressUpdate.Append(testFunction.GetDim());
                            progressUpdate.Append(" population ");
                            progressUpdate.Append(population);
                            progressUpdate.Append(" iterations ");
                            progressUpdate.Append(iterations);
                            progressUpdate.Append(" alpha ");
                            progressUpdate.Append(alpha);
                            progressUpdate.Append(" delta ");
                            progressUpdate.Append(delta);
                            
                            Console.WriteLine(progressUpdate.ToString());
                            
                            var dim = testFunction.GetDim();
                            var results = Enumerable.Range(0, n).AsParallel().Select(_ => new Aquila(
                                population,
                                dim,
                                alpha,
                                delta,
                                testFunction.GetUpperBounds(),
                                testFunction.GetLowerBounds(),
                                testFunction.Fitness,
                                iterations
                            ).Predict()).ToList();
                            
                            var (xBest, yBest) = results.MinBy(x => x.XbestFitness);
                            var yWorst = results.MaxBy(x => x.XbestFitness).XbestFitness;
                            
                            var xAvgs = Enumerable.Range(0, dim).Select(i => results.Select(x => x.Xbest[i]).Average()).ToList();
                            var xStdDevs = Enumerable.Range(0, dim)
                                .Select(i => results.Select(x => x.Xbest[i]).PopulationStandardDeviation()).ToList();
                            
                            var yAvg = results.Select(x => x.XbestFitness).Average();
                            var yStdDev = results.Select(x => x.XbestFitness).PopulationStandardDeviation();
                            
                            var logString = new StringBuilder("AO,");
                            logString.Append(testFunction.displayName);
                            logString.Append(',');
                            logString.Append(dim);
                            logString.Append(',');
                            logString.Append(alpha);
                            logString.Append(',');
                            logString.Append(delta);
                            logString.Append(',');
                            logString.Append(iterations);
                            logString.Append(',');
                            logString.Append(population);
                            logString.Append(",\"(");
                            logString.Append(string.Join(',', xBest));
                            logString.Append(")\",\"(");
                            logString.Append(string.Join(',', xStdDevs));
                            logString.Append(")\",\"(");
                            logString.Append(string.Join(',', xAvgs));
                            logString.Append(")\",");
                            logString.Append(yBest);
                            logString.Append(',');
                            logString.Append(yStdDev);
                            logString.Append(',');
                            logString.Append(yAvg);
                            logString.Append(',');
                            logString.Append(yWorst);
                            
                            Logging.LogText(logString.ToString());
                        }
                    }
                }
            }
        }

        Logging.Flush();
    }
}