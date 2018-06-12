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
//			NNet nn2X2 = new NNet(0)
//			{
//				Input = Vector.FromArray(new[] { 1.0, 0.5 }),
//				Weights = new[]
//				{
//					new Matrix().FromArray(new[,]
//					{
//						{ 0.9, 0.2 },
//						{ 0.3, 0.8 },
//					}),
//				}
//			};
//			nn2X2.FeedForward();
//			Console.WriteLine($"nn2x2 out: {nn2X2.Output}");

//			NNet nn3X3 = new NNet(1)
//			{
//				Input = Vector.FromArray(new[] { 0.9, 0.2, 0.8 }),
//				Weights = new[]
//				{
//					new Matrix().FromArray(new[,]
//					{
//						{ 0.9, 0.3, 0.4 },
//						{ 0.2, 0.8, 0.2 },
//						{ 0.1, 0.5, 0.6 },
//					}),
//					new Matrix().FromArray(new[,]
//					{
//						{ 0.3, 0.7, 0.5 },
//						{ 0.6, 0.5, 0.2 },
//						{ 0.8, 0.1, 0.9 },
//					})
//				}
//			};
//			nn3X3.FeedForward();
//			Console.WriteLine($"nn3x3 out: {nn3X3.Output}");

			NNet xor = new NNet(1)
			{
				Input = Vector.FromArray(new[] { 1.0, 0.0 }),
				Weights = new[]
				{
					new Matrix().FromArray(new[,]
					{
						{ 0.9, 0.2 },
						{ 0.3, 0.8 },
					}),
					new Matrix().FromArray(new[,]
					{
						{ 0.7, 0.6 },
						//{ 0.3 }, //, 0.8 },
					}),
				}
			};
			xor.FeedForward();
			Console.WriteLine($"XOR: {xor.Output}");
			Console.WriteLine(xor);
			xor.BackPropogate(new[] { 0.1 }); // Expect 0 (becomes 0.1).
			Console.WriteLine(xor);

			xor.FeedForward();
			Console.WriteLine($"XOR: {xor.Output}");
			for (int i = 0; i < 100000; i++)
			{
				xor.FeedForward();
				xor.BackPropogate(new[] { 0.9 }); // Expect 0 (becomes 0.1).
			}

			xor.FeedForward();
			Console.WriteLine($"XOR: {xor.Output}");
		}

		public class NNet
		{
			public NNet(int hiddenLayerCount)
			{
				HiddenLayerCount = hiddenLayerCount;
				HiddenLayers     = new Vector[hiddenLayerCount + 2];
			}

			public Vector Input
			{
				set => HiddenLayers[0] = value;
			}

			public int HiddenLayerCount { get; set; }

			// Rows is destination, Col is source.
			public Matrix[] Weights      { get; set; }
			public Vector[] HiddenLayers { get; }

			public Vector Output => HiddenLayers.Last();

			public void FeedForward()
			{
				for (int i = 0; i < HiddenLayerCount + 1; i++)
				{
					HiddenLayers[i + 1] = (Weights[i] * HiddenLayers[i]).ApplySigmoid();
				}
			}

			public void BackPropogate(double[] expectedOutput)
			{
				// double[] expected = expectedOutput;

				double[] errorArray = new double[expectedOutput.Length];
				for (int i = 0; i < Output.Count; i++)
					errorArray[i] = expectedOutput[i] - Output[i];

				for (int currentLayer = HiddenLayerCount + 1; currentLayer >= 1; currentLayer--)
				{
					// Retrieve weights for layer.
					IMatrix   w  = Weights[currentLayer - 1];
					double[,] dw = new double[w.RowCount, w.ColCount];

					double[] nextErrorArray = new double[HiddenLayers[currentLayer - 1].Count];

					for (int destNodeI = 0; destNodeI < HiddenLayers[currentLayer].Count; destNodeI++)
					{
						double valCurrentDest  = HiddenLayers[currentLayer][destNodeI];
						Vector valsSourceNodes = HiddenLayers[currentLayer - 1];

						double sigma = 0d;
						for (int sourceNodeI = 0; sourceNodeI < valsSourceNodes.Count; sourceNodeI++)
						{
							// Get the weight for this set.
							sigma += valsSourceNodes[sourceNodeI] * w[destNodeI, sourceNodeI];

							nextErrorArray[sourceNodeI] += errorArray[destNodeI] * w[destNodeI, sourceNodeI];
						}

						double sigmoid = Sigmoid(sigma);

						double total = -errorArray[destNodeI] * sigmoid * (1 - sigmoid) *
						               HiddenLayers[currentLayer][destNodeI];

						for (int sourceNodeI = 0; sourceNodeI < valsSourceNodes.Count; sourceNodeI++)
						{
							dw[destNodeI, sourceNodeI] += -(total * w[destNodeI, sourceNodeI] * 0.1); // TODO learnrate
						}
					}

					errorArray = nextErrorArray;

					Weights[currentLayer - 1] += new Matrix().FromArray(dw);
				}
			}

			public override string ToString()
			{
				return $"{string.Join<Vector>(Environment.NewLine, HiddenLayers)}";
			}
		}

		#region Extensions

//		// TODO Use DataStructures implementation once released.
//		private static Vector Mul(this Matrix m, Vector v)
//		{
//			Matrix res = m * new Matrix().FromArray(v.ToMultiArray());
//			return FromMatrix(res);
//		}
//
//		// TODO Use DataStructures implementation once released.
//		private static double[,] ToMultiArray(this Vector v)
//		{
//			double[,] res = new double[v.Count, 1];
//			for (int i = 0; i < v.Count; i++)
//			{
//				res[i, 0] = v[i];
//			}
//
//			return res;
//		}
//
//		// TODO Use DataStructures implementation once released.
//		private static Vector FromMatrix(IMatrix m)
//		{
//			double[] data = new double [m.RowCount];
//			for (int i = 0; i < m.RowCount; i++)
//			{
//				data[i] = m[i, 0];
//			}
//
//			return Vector.FromArray(data);
//		}

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