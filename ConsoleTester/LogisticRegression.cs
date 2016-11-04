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
            Matrix p = Sigmoid(X * theta);

            return p;
        }
    }
}
