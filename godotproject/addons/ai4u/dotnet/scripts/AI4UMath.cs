using System.Linq;

namespace ai4u.math;

public sealed class AI4UMath
{
    private AI4UMath()
    {

    }


    public static float Clip(float v, float start, float end)
    {
        if (v < start)
        {
            return start;
        }
        else if (v > end)
        {
            return end;
        }
        else
        {
            return v;
        }
    }

    public static int ArgMax(float[] values)
	{
		if (values.Length <= 0)
		{
			return -1;
		}
		if (values.Length == 1)
		{
			return 0;
		}
		int maxi = 0;
		for (int i = 1; i < values.Length; i++)
		{
			if (values[i] > values[maxi])
			{
				maxi = i;
			}
		}
		return maxi;
	}
}