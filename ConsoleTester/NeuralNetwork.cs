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

        /// <summary>
        /// Calculate the gradient of the Sigmoid function at z.
        /// </summary>
        /// <param name="z">The Matrix to calculate the gradient for each element.</param>
        /// <returns>The gradient of the Sigmoid function for each element of z.</returns>
        public static Matrix SigmoidGradient(Matrix z)
        {
            Matrix sg = LogisticRegression.Sigmoid(z);
            Matrix sg2 = 1.0 - sg;

            return Matrix.ElementMultiply(sg, sg2);
        }


        public static Tuple<double, Matrix> NNCostFunction(Matrix nn_parameters, int input_layer_size, int hidden_layer_size,
                                                    double[] labels, Matrix X, Matrix y, double lambda)
        {
            double costFunction = 0;
            int num_labels = labels.Length;
            List<Matrix> output_gradient = new List<Matrix>();

            Matrix Theta1 = Matrix.Reshape(nn_parameters, 0, hidden_layer_size, input_layer_size + 1);
            Matrix Theta2 = Matrix.Reshape(nn_parameters, (hidden_layer_size * (input_layer_size + 1)), num_labels, hidden_layer_size + 1);

            // y_matrix has the following attributes:
            // Rows: same as the number of rows in Y -- one for each example result.
            // Columns: one for each label.
            // Values: Each row consists of zeros, except for one, which matches the
            // value of y in that row to the index of the label. For example, if there
            // are three labels (3, 6, 8), and y contains 2 rows (8, 3), then y_matrix
            // would be:
            // 0 0 1
            // 1 0 0
            Matrix y_matrix = AssignLabels(y, labels);

            // Add ones to the X Matrix
            Matrix a1 = Matrix.AddIdentityColumn(X);

            Matrix z2 = a1 * Theta1.Transpose;
            Matrix a2 = LogisticRegression.Sigmoid(z2);
            a2 = Matrix.AddIdentityColumn(a2);

            Matrix z3 = a2 * Theta2.Transpose;
            Matrix a3 = LogisticRegression.Sigmoid(z3);

            Matrix log1 = Matrix.ElementLog(a3);
            Matrix log2 = Matrix.ElementLog(1 - a3);

            Matrix part1 = Matrix.ElementMultiply(-y_matrix, log1);
            Matrix part2 = Matrix.ElementMultiply((1 - y_matrix), log2);

            Matrix t0 = Theta1.RemoveColumn(0);
            Matrix t1 = Theta2.RemoveColumn(0);

            // Calculate regularization component
            double multiplier = lambda / (2 * X.Rows);
            double reg1 = Matrix.ElementPower(t0, 2).SumAllElements;
            double reg2 = Matrix.ElementPower(t1, 2).SumAllElements;
            double r = multiplier * (reg1 + reg2);

            // Calculate cost
            costFunction = (1.0 / X.Rows) * (part1 - part2).SumAllElements + r;


            // Back Propogation
            Matrix d3 = a3 - y_matrix;
            Matrix d2 = Matrix.ElementMultiply(
                            (t1.Transpose * d3.Transpose).Transpose,
                            SigmoidGradient(z2)
                        );

            Matrix Delta1 = d2.Transpose * a1;
            Matrix Delta2 = d3.Transpose * a2;

            Theta1 = Matrix.Join(new Matrix(t0.Rows, 1), t0, MatrixDimensions.Columns);
            Theta2 = Matrix.Join(new Matrix(t1.Rows, 1), t1, MatrixDimensions.Columns);

            double scale_value = lambda / X.Rows;
            Matrix Theta1_scaled = Theta1 * scale_value;
            Matrix Theta2_scaled = Theta2 * scale_value;

            Matrix Theta1_grad = ( (Delta1 / X.Rows) + Theta1_scaled ).Unrolled;
            Matrix Theta2_grad = ( (Delta2 / X.Rows) + Theta2_scaled ).Unrolled;

            return new Tuple<double, Matrix>(costFunction, Matrix.Join(Theta1_grad, Theta2_grad, MatrixDimensions.Rows));
        }

        private static Matrix AssignLabels(Matrix m, double[] labels)
        {
            Matrix result = new Matrix(m.Rows, labels.Length);

            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < labels.Length; j++)
                {
                    if (m[i,0] == labels[j])
                    {
                        result[i, j] = 1;
                        break;
                    }
                }
            }

            return result;
        }
    }
}
