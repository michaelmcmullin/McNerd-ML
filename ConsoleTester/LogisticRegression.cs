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
        /// <summary>
        /// Compute the Sigmoid (logistic) function of a given Matrix.
        /// </summary>
        /// <param name="z">The Matrix to compute the Sigmoid function of.</param>
        /// <returns>A Matrix translating the original values into numbers
        /// between 0 and 1.</returns>
        public static Matrix Sigmoid(Matrix z)
        {
            return 1 / (1 + Matrix.ElementExp(-z));
        }

        /// <summary>
        /// Predict if each element in a Matrix is 'true' or 'false'
        /// (actually '1' or '0').
        /// </summary>
        /// <param name="X">The original feature Matrix.</param>
        /// <param name="theta">The theta values to apply to each feature.</param>
        /// <returns>A Matrix of 1 or 0 if a classification is likely or unlikely.</returns>
        public static Matrix Predict(Matrix X, Matrix theta)
        {
            return Sigmoid(X * theta) >= 0.5;
        }

        /// <summary>
        /// Calculate the cost function for logistic regression.
        /// </summary>
        /// <param name="X">The original features Matrix.</param>
        /// <param name="y">The results Matrix.</param>
        /// <param name="theta">The theta values to apply to each feature.</param>
        /// <returns>The cost of using the given value of theta, and the gradient of
        /// the cost (useful for iterative minimization functions)</returns>
        public static Tuple<double, Matrix> CostFunction(Matrix X, Matrix y, Matrix theta)
        {
            double m = (double)X.Rows;
            Matrix h = Sigmoid(X * theta);  // Hypothesis
            Matrix ev = h - y;              // Error Vector

            double part1 = (-y.Transpose * Matrix.ElementLog(h)).SumAllElements;
            double part2 = ((1 - y).Transpose * Matrix.ElementLog(1 - h)).SumAllElements;

            double J = (1.0 / m) * (part1 - part2);
            Matrix grad = (1 / m) * (X.Transpose * (h - y));
            return Tuple.Create(J, grad);
        }


        /// <summary>
        /// Calculate the regularized cost function for logistic regression.
        /// </summary>
        /// <param name="X">The original features Matrix.</param>
        /// <param name="y">The results Matrix.</param>
        /// <param name="theta">The theta values to apply to each feature.</param>
        /// <param name="lambda">The regularization parameter which helps reduce overfitting.
        /// Note that using values that are too high will lead to underfitting.</param>
        /// <returns>The cost of using the given value of theta, and the gradient of
        /// the cost (useful for iterative minimization functions)</returns>
        public static Tuple<double, Matrix> CostFunction(Matrix X, Matrix y, Matrix theta, double lambda)
        {
            double m = (double)X.Rows;
            Matrix t = new Matrix(theta);
            Matrix h = Sigmoid(X * t);  // Hypothesis
            Matrix ev = h - y;              // Error Vector
            double part1 = (-y.Transpose * Matrix.ElementLog(h)).SumAllElements;
            double part2 = ((1 - y).Transpose * Matrix.ElementLog(1 - h)).SumAllElements;

            double J = (1.0 / m) * (part1 - part2);

            t[0, 0] = 0;
            double theta_sq = (t.Transpose * t).SumAllElements;

            J += ((lambda / (2.0 * m)) * theta_sq);
            Matrix grad = ((1 / m) * (X.Transpose * (h - y))) + ((lambda / m) * t);

            return Tuple.Create(J, grad);
        }

        /// <summary>
        /// Create multiple logistic regression classifiers, one for each class we're
        /// trying to categorize.
        /// </summary>
        /// <param name="X">The features Matrix (n x m).</param>
        /// <param name="y">The results Matrix (n x 1).</param>
        /// <param name="labels">The labels to classify against.</param>
        /// <param name="lambda">The regularization parameter.</param>
        /// <returns>A Matrix where each row is a learned set of parameters for that
        /// particular class.</returns>
        public static Matrix OneVsAll(Matrix X, Matrix y, double[] labels, double lambda, int maxIterations = 50)
        {
            int m = X.Rows;
            int n = X.Columns;
            X = Matrix.Join(Matrix.Ones(m, 1), X, MatrixDimensions.Columns);
            int numberOfLabels = labels.Length;

            Matrix all_theta = new Matrix(numberOfLabels, n + 1);

            for (int c = 0; c < numberOfLabels; c++)
            {
                Matrix initial_theta = new Matrix(n + 1, 1);
                int i = 0;
                Matrix new_theta = Minimize(CostFunction, X, y == labels[c], initial_theta, lambda, maxIterations, out i);
                all_theta.SetRow(c, new_theta.Transpose);
            }

            return all_theta;
        }

        /// <summary>
        /// The signature of a method that can be passed to the Minimize method.
        /// </summary>
        /// <param name="X">The features Matrix (n x m).</param>
        /// <param name="y">The results Matrix (n x 1).</param>
        /// <param name="theta">The initial theta values.</param>
        /// <param name="lambda">The regularization parameter.</param>
        /// <returns>The cost of using the given value of theta, and the gradient of
        /// the cost (useful for iterative minimization functions)</returns>
        public delegate Tuple<double, Matrix> MinimizeFunction(Matrix X, Matrix y, Matrix theta, double lambda);

        /// <summary>
        /// Minimize the cost function for an initial set of values.
        /// </summary>
        /// <param name="f">A function that calculates the cost, and a vector
        /// of partial derivatives.</param>
        /// <param name="Features">The features being assessed for each label.</param>
        /// <param name="y">A truth table containing '1' if it matches the label
        /// being investigated, or '0' if it's not a match.</param>
        /// <param name="theta">Initial values for theta.</param>
        /// <param name="lambda">The regularization parameter.</param>
        /// <param name="maxIterations">The maximum number of iterations to perform
        /// before stopping.</param>
        /// <param name="i">The final number of iterations used to find a result.</param>
        /// <returns>The solution for theta for a given set of labels.</returns>
        public static Matrix Minimize(MinimizeFunction f, Matrix Features, Matrix y, Matrix theta, double lambda, int maxIterations, out int i)
        {
            int length = maxIterations > 0 ? maxIterations : 100;

            // Most of the below is adapted from fmincg.m by Carl Edward Rasmussen.
            // Original Copyright notice:
            // -------------------------------------------------------------------------
            // Copyright(C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002 - 02 - 13
            //           
            // (C)Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
            //
            // Permission is granted for anyone to copy, use, or modify these
            // programs and accompanying documents for purposes of research or
            // education, provided this copyright notice is retained, and note is
            // made of any changes that have been made.
            //
            // These programs and documents are distributed without any warranty,
            // express or implied.As the programs were written for research
            // purposes only, they have not been tested to the degree that would be
            // advisable in any important application.All use of these programs is
            // entirely at the user's own risk.
            // -------------------------------------------------------------------------
            // NOTE: Original code was written in Octave, while here it's obviously been
            // re-written in C#. There are likely a few differences and errors with this
            // implementation, which will hopefully be ironed out in time. These are
            // entirely my own fault, and not the original author's.

            double RHO = 0.01;                            // a bunch of constants for line searches
            double SIG = 0.5;       // RHO and SIG are the constants in the Wolfe - Powell conditions
            double INT = 0.1;    // don't reevaluate within 0.1 of the limit of the current bracket
            double EXT = 3.0;                    // extrapolate maximum 3 times the current bracket
            double MAX = 20;                         // max 20 function evaluations per line search
            double RATIO = 100;                                      // maximum allowed slope ratio

            double red = 1;

            i = 0;                                            // zero the run length counter
            bool ls_failed = false;                             // no previous line search has failed
            // fX = [];

            //[f1 df1] = eval(argstr);                      // get function value and gradient
            Tuple<double, Matrix> cost1 = f(Features, y, theta, lambda);
            double f1 = cost1.Item1;
            Matrix df1 = cost1.Item2;

            i = i + (length < 0 ? 1 : 0);                                            // count epochs?!
            Matrix s = -df1;                                        // search direction is steepest
            double d1 = (-Matrix.MultiplyTransposeBy(s))[0, 0];                                     // this is the slope
            double z1 = red / (1.0 - d1);                                  // initial step is red/(|s|+1)
            double z2, A, B;

            while (i < Math.Abs(length))                                      // while not finished
            {
                i = i + (length > 0 ? 1 : 0);                                      // count iterations?!

                Matrix theta0 = new Matrix(theta); double f0 = f1; Matrix df0 = new Matrix(df1);                   // make a copy of current values
                theta = theta + z1 * s;                                             // begin line search
                Tuple<double, Matrix> cost2 = f(Features, y, theta, lambda);
                double f2 = cost2.Item1;
                Matrix df2 = new Matrix(cost2.Item2);

                //[f2 df2] = eval(argstr);
                i = i + (length < 0 ? 1 : 0);                                          // count epochs?!
                double d2 = MultiplyFirstRowTransposeBy(df2, s);
                double f3 = f1; double d3 = d1; double z3 = -z1;             // initialize point 3 equal to point 1
                double M = 0;

                M = (length > 0) ? MAX : Math.Min(MAX, -length - i);

                bool success = false; double limit = -1;                     // initialize quantities
                while (true)
                {
                    while (((f2 > f1 + z1 * RHO * d1) || (d2 > -SIG * d1)) && (M > 0))
                    {
                        limit = z1;                                         // tighten the bracket
                        z2 = 0;
                        if (f2 > f1)
                            z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);                 // quadratic fit
                        else
                        {
                            A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);                                 // cubic fit
                            B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                            z2 = (Math.Sqrt(B * B - A * d2 * z3 * z3) - B) / A;       // numerical error possible -ok!
                        }

                        if ((double.IsNaN(z2)) || (double.IsInfinity(z2)))
                        {
                            z2 = z3 / 2;                  // if we had a numerical problem then bisect
                        }
                        z2 = Math.Max(Math.Min(z2, INT * z3), (1 - INT) * z3);  // don't accept too close to limits
                        z1 = z1 + z2;                                           // update the step
                        theta = theta + z2 * s;
                        // [f2 df2] = eval(argstr);
                        cost2 = f(Features, y, theta, lambda);
                        f2 = cost2.Item1;
                        df2 = new Matrix(cost2.Item2);

                        M = M - 1; i = i + (length<0 ? 1 : 0);                           // count epochs?!
                        //d2 = (df2.Transpose * s)[0, 0];
                        d2 = MultiplyFirstRowTransposeBy(df2, s); // (Matrix.MultiplyTransposeBy(df2, s))[0, 0];
                        z3 = z3-z2;                    // z3 is now relative to the location of z2
                    }
                    if ((f2 > (f1 + z1 * RHO * d1)) || (d2 > (-SIG * d1)))
                        break;                                                // this is a failure
                    else if (d2 > (SIG * d1))
                    {
                        success = true;
                        break;                                             // success
                    }
                    else if (M == 0)
                      break;                                                          // failure

                    A = 6*(f2-f3)/z3+3*(d2+d3);                      // make cubic extrapolation
                    B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                    z2 = -d2* z3*z3/(B+Math.Sqrt(B* B-A* d2*z3* z3));        // num.error possible - ok!
                    //if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0        // num prob or wrong sign?
                    if (double.IsNaN(z2) || double.IsInfinity(z2) || z2 < 0)        // num prob or wrong sign?
                    {
                        if (limit < -0.5)                               // if we have no upper limit
                            z2 = z1 * (EXT - 1);                 // the extrapolate the maximum amount
                        else
                            z2 = (limit - z1) / 2;                                   // otherwise bisect
                    }
                    else if((limit > -0.5) && (z2+z1 > limit))         // extraplation beyond max?
                        z2 = (limit - z1) / 2;                                               // bisect
                    else if((limit< -0.5) && (z2+z1 > z1* EXT))      // extrapolation beyond limit
                        z2 = z1 * (EXT - 1.0);                           // set to extrapolation limit
                    else if (z2< -z3* INT)
                        z2 = -z3 * INT;
                    else if((limit > -0.5) && (z2< (limit-z1)*(1.0-INT)))  // too close to limit?
                        z2 = (limit-z1)*(1.0-INT);
                    f3 = f2; d3 = d2; z3 = -z2;                  // set point 3 equal to point 2
                    z1 = z1 + z2; theta = theta + z2* s;                      // update current estimates
                    //[f2 df2] = eval(argstr);
                    cost2 = f(Features, y, theta, lambda);
                    f2 = cost2.Item1;
                    df2 = new Matrix(cost2.Item2);

                    M = M - 1; i = i + (length < 0 ? 1 : 0);                             // count epochs?!
                    d2 = MultiplyFirstRowTransposeBy(df2, s);
                }                                                   // end of line search



                if (success)                                         // if line search succeeded
                {
                    f1 = f2; //fX = [fX' f1]';
                    //fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
                    s = (df2.Transpose * df2 - df1.Transpose * df2)[0,0] / (df1.Transpose * df1)[0,0] * s - df2;     // Polack-Ribiere direction
                    Matrix tmp = new Matrix(df1); df1 = df2; df2 = tmp;                        // swap derivatives
                    //d2 = (df1.Transpose * s)[0, 0];
                    d2 = (Matrix.MultiplyTransposeBy(df1, s))[0, 0];
                    if (d2 > 0)                                      // new slope must be negative
                    {
                        s = -df1;                              // otherwise use steepest direction
                        //d2 = (-s.Transpose * s)[0, 0];
                        d2 = MultiplyFirstRowTransposeBy(s, true); // d2 = (-Matrix.MultiplyTransposeBy(s))[0, 0];
                    }
                    double TEST = d2-double.MinValue;
                    z1 = z1 * Math.Min(RATIO, d1 / (d2 - double.Epsilon));          // slope ratio but max RATIO
                    d1 = d2;
                    ls_failed = false;                             // this line search did not fail
                }
                else
                {
                    theta = new Matrix(theta0); f1 = f0; df1 = df0;  // restore point from before failed line search
                    if (ls_failed || i > Math.Abs(length))         // line search failed twice in a row
                        break;                             // or we ran out of time, so we give up

                    Matrix tmp = new Matrix(df1); df1 = df2; df2 = tmp;                         // swap derivatives
                    s = -df1;                                                    // try steepest
                    d1 = MultiplyFirstRowTransposeBy(s, true);
                    z1 = 1 / (1 - d1);
                    ls_failed = true;                                    //this line search failed
                }
            }

            return theta;
        }
        private static double MultiplyFirstRowTransposeBy(Matrix m1, bool negate = false)
        {
            double result = 0;

            if (negate)
            {
                for (int i = 0; i < m1.Rows; i++)
                {
                    double nextNumber = m1[i, 0];
                    result += -nextNumber * nextNumber;
                }
            }
            else
            {
                for (int i = 0; i < m1.Rows; i++)
                {
                    double nextNumber = m1[i, 0];
                    result += nextNumber * nextNumber;
                }
            }

            return result;
        }
        private static double MultiplyFirstRowTransposeBy(Matrix m1, Matrix m2, bool negate = false)
        {
            double result = 0;

            if (negate)
            {
                for (int i = 0; i < m1.Rows; i++)
                    result += -m1[i, 0] * m2[i, 0];
            }
            else
            {
                for (int i = 0; i < m1.Rows; i++)
                    result += m1[i, 0] * m2[i, 0];
            }

            return result;
        }

        /// <summary>
        /// Using a trained classifier (all_theta), predict the index of each classifier
        /// that applies to each row of X.
        /// </summary>
        /// <param name="all_theta">A Matrix where each row is a trained logistic regression
        /// for a single output classification.</param>
        /// <param name="X">A Matrix of example rows.</param>
        /// <returns>A Matrix (column vector, m * 1) containing the zero-based indices of
        /// the most probable classification prediction for each input row in X.</returns>
        public static Matrix PredictOneVsAll(Matrix all_theta, Matrix X)
        {
            Matrix h = Sigmoid(Matrix.MultiplyByTranspose(all_theta, Matrix.AddIdentityColumn(X)));
            return Matrix.MaxIndex(h).Transpose;
        }
    }
}
