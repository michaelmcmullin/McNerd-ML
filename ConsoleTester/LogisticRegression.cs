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
    }
}
