using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using McNerd.MachineLearning.LinearAlgebra;

namespace ConsoleTester
{
    /// <summary>
    /// This console app is just for playing around with the code.
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m3 = new Matrix(new double[,] {
                { 7.0, 8.0 },
                { 9.0, 10.0 },
                { 11.0, 12.0 }
            });
            Matrix m4 = new Matrix(400);
            Matrix m5 = new Matrix(400);

            // A quick and dirty test to try out timing.
            for (int i=0; i<10; i++)
            {
                Matrix mtest = m4 * m5;
                double x = mtest[1, 1];
            }

            Console.ReadLine();

            // Test 1: approx 500ms (addition - 1000000 iterations, 2x3 + 2x3)
            // Test 2: approx 1,150ms (multiplication - 1000000 iterations, 2x3 * 3x2)
            // Test 3: approx 45,880ms (multiplication - 10 iterations, 400x400 * 400x400)
            //         approx 33,374ms (changing multidimensional array to jagged array)
            //         approx 32,471ms (changing to single-dimensional array)
            //         approx 22,000ms (changed multiplication indexing method for Matrix 1)
            //         approx 7,881ms (changed multiplication indexing method for Matrix 2)
        }
    }
}
