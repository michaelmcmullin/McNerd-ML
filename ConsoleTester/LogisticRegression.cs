using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using McNerd.MachineLearning.LinearAlgebra;

namespace ConsoleTester
{
    class LogisticRegression
    {
        public static Matrix Sigmoid(Matrix z)
        {
            return 1 / (1 + Matrix.ElementExp(-z));
        }

        public static Matrix Predict(Matrix X, Matrix theta)
        {
            return Sigmoid(X * theta) >= 0.5;
        }

        public static double CostFunction(Matrix X, Matrix y, Matrix theta)
        {
            Matrix h = Sigmoid(X * theta);  // Hypothesis
            Matrix ev = h - y;              // Error Vector

            double part1 = (-y.Transpose * Matrix.ElementLog(h)).SumAllElements;
            double part2 = ((1 - y).Transpose * Matrix.ElementLog(1 - h)).SumAllElements;

            double output = (1.0 / (double)X.Rows) * (part1 - part2);

            return output;
        }

        public static double CostFunction(Matrix X, Matrix y, Matrix theta, double lambda)
        {
            double output = CostFunction(X, y, theta);

            theta[0, 0] = 0;
            double theta_sq = (theta.Transpose * theta).SumAllElements;

            output += ((lambda / (2.0 * (double)X.Rows)) * theta_sq);

            return output;
        }

        public static Matrix OneVsAll(Matrix X, Matrix y, int numberOfLabels, double lambda)
        {
            int m = X.Rows;
            int n = X.Columns;
            X = Matrix.Join(Matrix.Ones(m, 1), X, MatrixDimensions.Columns);
            
            Matrix all_theta = new Matrix(numberOfLabels, n+1);

            for (int c=0; c < numberOfLabels; c++)
            {

            }

            return all_theta;
        }

    }
}
