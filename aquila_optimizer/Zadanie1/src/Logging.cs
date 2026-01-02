namespace aquila;

public static class Logging
{
    static StreamWriter _writer = new ("raport.log");
    
    public static void LogText(string msg)
    {
        _writer.WriteLine(msg);
    }

    public static void LogBounds(List<double> upperBounds, List<double> lowerBounds)
    {
        string msg = "";
        for (int i = 0; i < upperBounds.Count; i++)
        {
            msg += lowerBounds[i] + "," + upperBounds[i] + ",";
        }
        msg = msg.TrimEnd(',');
        LogText(msg);
    }

    public static void LogX(int iteration, List<double> x)
    {
        string msg = iteration.ToString() + ",";
        foreach (var xi in x)
        {
            msg += xi + ",";
        }
        msg = msg.TrimEnd(',');
        LogText(msg);
    }

    public static void Flush()
    {
        _writer.Flush();
    }
}