using System.Linq;
using DataStructures.Matrices;
using DataStructures.Vectors;

namespace MLDemos
{
	/// <summary>
	/// An unbiased neural network.
	/// </summary>
	public class NeuralNetwork
	{
		/// <summary>
		/// The number of hidden layers in the neural network.
		/// </summary>
		public int HiddenLayerCount;

		/// <summary>
		/// The number of layers in the neural network. Includes all the hidden layers and the in- and output layers.
		/// </summary>
		public int LayerCount => HiddenLayerCount + 2;

		/// <summary>
		/// The number of weight sets (i.e., the number of layer transistions).
		/// </summary>
		public int WeightCount => HiddenLayerCount + 1;

		private Matrix[] _weights;

		private Vector[] _previousRun;

		public NeuralNetwork(Matrix[] weights)
		{
			_weights = weights;

			HiddenLayerCount = _weights.Length - 1;
			_previousRun     = new Vector[LayerCount];
		}

		/// <summary>
		/// Make the neural network produce output.
		/// </summary>
		/// <param name="input">The input to have the neural network interpret.</param>
		/// <returns>The input, interpreted.</returns>
		public Vector FeedForward(Vector input)
		{
			Vector previousOutput = input;
			_previousRun[0] = input;
			int nextLayerIndex = 1;

			while (nextLayerIndex < LayerCount)
			{
				previousOutput = (_weights[nextLayerIndex - 1] * previousOutput).ApplySigmoid();

				_previousRun[nextLayerIndex] = previousOutput;
				nextLayerIndex++;
			}

			return previousOutput;
		}

		public void Epoch(double[] input, double[] expectedOutput)
		{
			Epoch(Vector.Build(input.ToList()), Vector.Build(expectedOutput.ToList()));
		}

		public void Epoch(Vector input, Vector expectedOutput)
		{
			Vector actualOutput = FeedForward(input);

			Vector   error = expectedOutput - actualOutput;
			double[] nextError;

			// Loop over all the weight layers that need updating.
			for (int i = WeightCount - 1; i >= 0; i--)
			{
				Matrix    weights      = _weights[i];
				double[,] deltaWeights = new double[weights.RowCount, weights.ColCount];

				Vector sourceNodes = _previousRun[i];
				Vector destNodes   = _previousRun[i + 1];

				nextError = new double[sourceNodes.Count];

				// Loop over all the destination nodes in the current layer.
				for (int j = 0; j < destNodes.Length; j++)
				{
					double sigma = 0d;
					for (int k = 0; k < sourceNodes.Count; k++)
					{
						sigma += sourceNodes[k] * weights[j, k];

						nextError[k] += error[j] * weights[j, k];
					}

					double sigmoid = Program.Sigmoid(sigma);

					double total = -error[j] * sigmoid * (1 - sigmoid) * destNodes[j];

					for (int k = 0; k < sourceNodes.Count; k++)
					{
						deltaWeights[j, k] += -(total * weights[j, k] * 0.1);
					}
				}

				error = Vector.Build(nextError.ToList());

				_weights[i] += new Matrix().FromArray(deltaWeights);
			}
		}
	}
}