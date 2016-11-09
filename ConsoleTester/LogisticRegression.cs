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

        public static Tuple<double, Matrix> CostFunction(Matrix X, Matrix y, Matrix theta)
        {
            double m = (double)X.Rows;
            Matrix h = Sigmoid(X * theta);  // Hypothesis
            Matrix ev = h - y;              // Error Vector

            double part1 = (-y.Transpose * Matrix.ElementLog(h)).SumAllElements;
            double part2 = ((1 - y).Transpose * Matrix.ElementLog(1 - h)).SumAllElements;

            double J = (1.0 / m) * (part1 - part2);
            Matrix grad = (1 / m) * (X.Transpose*(h-y));
            return Tuple.Create(J, grad);
        }

        public static Tuple<double, Matrix> CostFunction(Matrix X, Matrix y, Matrix theta, double lambda)
        {
            double m = (double)X.Rows;
            Matrix h = Sigmoid(X * theta);  // Hypothesis
            Matrix ev = h - y;              // Error Vector
            double part1 = (-y.Transpose * Matrix.ElementLog(h)).SumAllElements;
            double part2 = ((1 - y).Transpose * Matrix.ElementLog(1 - h)).SumAllElements;

            double J = (1.0 / m) * (part1 - part2);

            theta[0, 0] = 0;
            double theta_sq = (theta.Transpose * theta).SumAllElements;

            J += ((lambda / (2.0 * m)) * theta_sq);
            Matrix grad = ((1 / m) * (X.Transpose*(h-y))) + ((lambda/m) * theta);

            return Tuple.Create(J, grad);
        }

        public static Matrix OneVsAll(Matrix X, Matrix y, int numberOfLabels, double lambda)
        {
            int m = X.Rows;
            int n = X.Columns;
            X = Matrix.Join(Matrix.Ones(m, 1), X, MatrixDimensions.Columns);
            
            Matrix all_theta = new Matrix(numberOfLabels, n+1);

            for (int c=0; c < numberOfLabels; c++)
            {
                Matrix initial_theta = new Matrix(n+1, 1);
            }

            return all_theta;
        }

        public static Matrix Minimize(Matrix X, Matrix y, Matrix theta, double lambda)
        {
            Matrix output = Matrix.Ones(1, theta.Rows);

            return output;
        }
    }
}
