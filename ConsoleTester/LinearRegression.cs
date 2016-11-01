using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using McNerd.MachineLearning.LinearAlgebra;

namespace ConsoleTester
{
    /// <summary>
    /// Testing out linear regression techniques
    /// </summary>
    /// <remarks>Eventually, these techniques will be incorporated into the main McNerd.MachineLearning
    /// namespace. They are being tested here to ensure they're working. As the number of methods increases,
    /// some sort of logical structure and reuse possibilities will hopefully present itself. That should
    /// lead to a more robust class library.</remarks>
    class LinearRegression
    {
        /// <summary>
        /// Compute the cost for linear regression with multiple variables.
        /// </summary>
        /// <param name="X">Input data, size m*n</param>
        /// <param name="y">Output results, size m*1</param>
        /// <param name="theta">Coefficients of X to test</param>
        /// <returns>A value representing the accuracy of using theta for our hypothesis function.
        /// The lower the result, the better the fit.</returns>
        public static double ComputeCost(Matrix X, Matrix y, Matrix theta)
        {
            double result = 0;
            double m = y.Rows;

            // Check inputs
            if (X == null || y == null || theta == null)
                throw new ArgumentNullException("ComputeCost requires that matrices are not null.");
            if ((X.Rows != y.Rows) || (X.Columns != theta.Rows))
                throw new InvalidMatrixDimensionsException("ComputeCost cannot work with matrices of these dimensions.");

            Matrix m1 = X * theta;
            Matrix m2 = m1 - y;
            Matrix m3 = Matrix.ElementPower(m2, 2);
            Matrix sumValues = Matrix.Sum(m3);

            result = (1.0 / (2.0 * m)) * sumValues[0, 0];

            return result;
        }

        /// <summary>
        /// Iteratively improve the value of theta (co-efficients in the hypothesis) through gradient descent.
        /// </summary>
        /// <param name="X">Input data, size m*n</param>
        /// <param name="y">Output results, size m*1</param>
        /// <param name="theta">Initial coefficients of X to improve on</param>
        /// <param name="alpha">The learning rate: too low, and it could take a long time; too high
        /// and it may never converge.</param>
        /// <param name="iterations">The number of iterations to try.</param>
        /// <returns>An n*1 Matrix of optimal co-efficients to use for our hypothesis function.</returns>
        public static Matrix GradientDescent(Matrix X, Matrix y, Matrix theta, double alpha, double iterations)
        {
            double m = y.Rows;

            // Check inputs
            if (X == null || y == null || theta == null)
                throw new ArgumentNullException("GradientDescent requires that matrices are not null.");
            if ((X.Rows != y.Rows) || (X.Columns != theta.Rows))
                throw new InvalidMatrixDimensionsException("GradientDescent cannot work with matrices of these dimensions.");

            for (int i=0; i<iterations; i++)
            {
                Matrix hypothesis = X * theta;
                Matrix errorVector = hypothesis - y;

                Matrix theta_change = (1 / m) * (alpha * (X.Transpose * errorVector));
                theta = theta - theta_change;
            }

            return theta;
        }


        public static Matrix FeatureNormalization(Matrix X)
        {
            Matrix mu = Matrix.Mean(X);
            Matrix sigma = Matrix.StandardDeviation(X);
            int m = X.Rows;

            Matrix mu_matrix = Matrix.Ones(m, 1) * mu;
            Matrix sigma_matrix = Matrix.Ones(m, 1) * sigma;
            Matrix X_norm = Matrix.ElementDivide(X - mu_matrix, sigma_matrix);

            return X_norm;
        }

        /// <summary>
        /// Find the value of theta (co-efficients in the hypothesis) without iteration.
        /// </summary>
        /// <param name="X">Input data, size m*n</param>
        /// <param name="y">Output results, size m*1</param>
        /// <returns>An n*1 Matrix of optimal co-efficients to use for our hypothesis function.</returns>
        public static Matrix NormalEquation(Matrix X, Matrix y)
        {
            Matrix output = Matrix.MultiplyTransposeBy(X).Inverse;
            output = output * (Matrix.MultiplyTransposeBy(X, y));
            return output;
        }
    }
}
