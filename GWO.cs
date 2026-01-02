using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GWO
{
    internal class GWOptimizer : IOptimizationAlgorithm
    {
        private string _name = "GreyWolfOptimizer";
        private double[] _xBest;
        private double _fBest;
        private int _numberOfEvaluationFitnessFunction;
        private List<double> _iterationHistory = new List<double>();

        public string Name
        {
            get { return Name; }
            set { _name = "GreyWolfOptimizer"; }
        }

        public double[] XBest
        {
            get { return _xBest; }
            set { _xBest = value; }
        }

        public double FBest
        {
            get { return _fBest; }
            set { _fBest = value; }
        }

        public int NumberOfEvaluationFitnessFunction
        {
            get { return _numberOfEvaluationFitnessFunction; }
            set { _numberOfEvaluationFitnessFunction = value; }
        }

        public List<double> IterationHistory
        {
            get { return _iterationHistory; }
        }

        int wolves; // number of search agents
        int maxIter; // maximum number of iterations
        double lb; // lower bound
        double ub; // upper bound
        int dim; // dimension of the problem

        public double Solve()
        {
            return FBest;
        }

        public void Optimize(
            Func<double[], double> fitnessFunction,
            int wolves,
            int maxIter,
            double lb,
            double ub,
            int dim,
            double aCoeff = 1,
            double cCoeff = 3
        )
        {
            this.wolves = wolves;
            this.maxIter = maxIter;
            this.lb = lb;
            this.ub = ub;
            this.dim = dim;

            NumberOfEvaluationFitnessFunction = 0;
            FBest = double.MaxValue;
            XBest = new double[dim];
            _iterationHistory = new List<double>();

            // Initialize the positions of search agents
            double[][] positions = new double[wolves][];
            double[] fitness = new double[wolves];
            Random rand = new Random(); // Random seed
            for (int i = 0; i < wolves; i++)
            {
                positions[i] = new double[dim];
                for (int j = 0; j < dim; j++)
                {
                    positions[i][j] = lb + rand.NextDouble() * (ub - lb);
                }
                fitness[i] = fitnessFunction(positions[i]);
                NumberOfEvaluationFitnessFunction++;
            }
            double alphaScore = double.MaxValue;
            double betaScore = double.MaxValue;
            double deltaScore = double.MaxValue;

            double[] alpha = new double[dim];
            double[] beta = new double[dim];
            double[] delta = new double[dim];

            // Main loop
            for (int iter = 0; iter < maxIter; iter++)
            {
                // Update alpha beta and delta
                for (int i = 0; i < wolves; i++)
                {
                    if (fitness[i] < alphaScore)
                    {
                        deltaScore = betaScore;
                        Array.Copy(beta, delta, dim);

                        betaScore = alphaScore;
                        Array.Copy(alpha, beta, dim);

                        alphaScore = fitness[i];
                        Array.Copy(positions[i], alpha, dim);
                    }
                    else if (fitness[i] < betaScore)
                    {
                        deltaScore = betaScore;
                        Array.Copy(beta, delta, dim);

                        betaScore = fitness[i];
                        Array.Copy(positions[i], beta, dim);
                    }
                    else if (fitness[i] < deltaScore)
                    {
                        deltaScore = fitness[i];
                        Array.Copy(positions[i], delta, dim);
                    }
                }

                double a = aCoeff - iter * (aCoeff / maxIter);

                // Update the positions of search agents
                for (int i = 0; i < wolves; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        double r1 = rand.NextDouble();
                        double r2 = rand.NextDouble();
                        double A1 = 2.0 * a * r1 - a;
                        double C1 = cCoeff * r2;
                        double D_alpha = Math.Abs(C1 * alpha[j] - positions[i][j]);
                        double X1 = alpha[j] - A1 * D_alpha;

                        r1 = rand.NextDouble();
                        r2 = rand.NextDouble();
                        double A2 = 2.0 * a * r1 - a;
                        double C2 = cCoeff * r2;
                        double D_beta = Math.Abs(C2 * beta[j] - positions[i][j]);
                        double X2 = beta[j] - A2 * D_beta;

                        r1 = rand.NextDouble();
                        r2 = rand.NextDouble();
                        double A3 = 2.0 * a * r1 - a;
                        double C3 = cCoeff * r2;
                        double D_delta = Math.Abs(C3 * delta[j] - positions[i][j]);
                        double X3 = delta[j] - A3 * D_delta;

                        positions[i][j] = (X1 + X2 + X3) / 3.0;

                        // Boundary check with reflection
                        if (positions[i][j] < lb)
                            positions[i][j] = lb + rand.NextDouble() * (ub - lb) * 0.1;
                        if (positions[i][j] > ub)
                            positions[i][j] = ub - rand.NextDouble() * (ub - lb) * 0.1;
                    }
                    fitness[i] = fitnessFunction(positions[i]);
                    NumberOfEvaluationFitnessFunction++;
                }

                // Zapisz najlepszy wynik dla tej iteracji
                _iterationHistory.Add(alphaScore);

                //if (iter % 100 == 0)
                //    Console.WriteLine($"Iteration {iter}, Best Fitness: {alphaScore}");
            }
            //Console.WriteLine($"Najlepsza wartość funkcji: {alphaScore}");
            //Console.WriteLine($"Najlepsze rozwiązanie (alfa): [{string.Join(", ", alpha)}]");

            XBest = alpha;
            FBest = alphaScore;
        }
    }
}
