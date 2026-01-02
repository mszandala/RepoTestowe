using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace aquila.Models;

public class Solution
{
    public List<double> Position { get; set; }
    public double Fitness { get; set; }

    public Solution(List<double> position, double fitness)
    {
        Position = position;
        Fitness = fitness;
    }

    public override string ToString()
    {
        string pos = string.Join(", ", Position.Select(p => p.ToString("F4")));
        return $"Fitness = {Fitness:F6}, Position = [{pos}]";
    }
}

