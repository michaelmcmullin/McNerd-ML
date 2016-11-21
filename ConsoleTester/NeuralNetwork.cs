using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using McNerd.MachineLearning.LinearAlgebra;

namespace ConsoleTester
{
    class NeuralNetwork
    {
        /// <summary>
        /// Predict the index of each classifer that applies to each row of X using trained
        /// weights of a neural network.
        /// </summary>
        /// <param name="theta_1">The first set of trained weights between the input layer
        /// and the hidden layer.</param>
        /// <param name="theta_2">The second set of trained weights between the hidden layer
        /// and the output layer.</param>
        /// <param name="X">A Matrix of example rows.</param>
        /// <returns>A Matrix (column vector, m * 1) containing the zero-based indices of
        /// the most probable classification prediction for each input row in X.</returns>
        /// <remarks>This version only has one hidden layer, which will suffice for many
        /// problems. However, it should probably be updated to allow for an arbitrary
        /// number of layers.</remarks>
        public static Matrix Predict(Matrix theta_1, Matrix theta_2, Matrix X)
        {
            Matrix A1 = Matrix.AddIdentityColumn(X);

            // Calculate the second (hidden) layer.
            Matrix Z2 = Matrix.MultiplyByTranspose(theta_1, A1);
            Matrix A2 = LogisticRegression.Sigmoid(Z2);
            A2 = Matrix.AddIdentityColumn(A2.Transpose);

            // Calculate 3rd layer (output)
            Matrix Z3 = Matrix.MultiplyByTranspose(theta_2, A2);
            Matrix A3 = LogisticRegression.Sigmoid(Z3);

            return Matrix.MaxIndex(A3).Transpose;
        }

        public static Matrix SigmoidGradient(Matrix z)
        {
            Matrix sg = LogisticRegression.Sigmoid(z);
            Matrix sg2 = 1.0 - sg;

            return Matrix.ElementMultiply(sg, sg2);
        }
    }
}
