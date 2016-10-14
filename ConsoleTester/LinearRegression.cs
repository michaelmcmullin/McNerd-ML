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
    }
}
