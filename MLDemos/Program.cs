using System;
using System.Linq;
using DataStructures.Matrices;
using DataStructures.Vectors;

namespace MLDemos
{
	public static class Program
	{
		public static void Main(string[] args)
		{
			NNet nn2X2 = new NNet(1)
			{
				Input = Vector.FromArray(new[] { 1.0, 0.5 }),
				Weights = new[]
				{
					new Matrix().FromArray(new[,]
					{
						{ 0.9, 0.2 },
						{ 0.3, 0.8 },
					}),
				}
			};
			nn2X2.FeedForward();
			Console.WriteLine($"nn2x2 out: {nn2X2.Output}");

			NNet nn3X3 = new NNet(2)
			{
				Input = Vector.FromArray(new[] { 0.9, 0.2, 0.8 }),
				Weights = new[]
				{
					new Matrix().FromArray(new[,]
					{
						{ 0.9, 0.3, 0.4 },
						{ 0.2, 0.8, 0.2 },
						{ 0.1, 0.5, 0.6 },
					}),
					new Matrix().FromArray(new[,]
					{
						{ 0.3, 0.7, 0.5 },
						{ 0.6, 0.5, 0.2 },
						{ 0.8, 0.1, 0.9 },
					})
				}
			};
			nn3X3.FeedForward();
			Console.WriteLine($"nn3x3 out: {nn3X3.Output}");
		}

		public class NNet
		{
			public NNet(int hiddenLayerCount)
			{
				HiddenLayerCount = hiddenLayerCount;
				HiddenLayers     = new Vector[hiddenLayerCount + 1];
			}

			public Vector Input
			{
				set => HiddenLayers[0] = value;
			}

			public int      HiddenLayerCount { get; set; }
			public Matrix[] Weights          { get; set; }
			public Vector[] HiddenLayers     { get; }

			public Vector Output => HiddenLayers.Last();

			public void FeedForward()
			{
				for (int i = 0; i < HiddenLayerCount; i++)
				{
					HiddenLayers[i + 1] = Weights[i].Mul(HiddenLayers[i]).ApplySigmoid();
				}
			}
		}

		#region Extensions

		// TODO Use DataStructures implementation once released.
		private static Vector Mul(this Matrix m, Vector v)
		{
			Matrix res = m * new Matrix().FromArray(v.ToMultiArray());
			return FromMatrix(res);
		}

		// TODO Use DataStructures implementation once released.
		private static double[,] ToMultiArray(this Vector v)
		{
			double[,] res = new double[v.Count, 1];
			for (int i = 0; i < v.Count; i++)
			{
				res[i, 0] = v[i];
			}

			return res;
		}

		// TODO Use DataStructures implementation once released.
		private static Vector FromMatrix(IMatrix m)
		{
			double[] data = new double [m.RowCount];
			for (int i = 0; i < m.RowCount; i++)
			{
				data[i] = m[i, 0];
			}

			return Vector.FromArray(data);
		}

		// TODO Use DataStructures implementation once released.
		private static Vector ApplySigmoid(this Vector v)
		{
			double[] data = new double[v.Count];
			for (int i = 0; i < v.Count; i++)
			{
				data[i] = Sigmoid(v[i]);
			}

			return Vector.FromArray(data);
		}

		// TODO Use DataStructures implementation once released.
		private static double Sigmoid(double d)
		{
			return 1d / (1 + Math.Pow(Math.E, -d));
		}

		#endregion
	}
}