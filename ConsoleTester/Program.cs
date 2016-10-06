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

            // A quick and dirty test to try out timing.
            for (int i=0; i<1000000; i++)
            {
                Matrix m3 = m1 + m2;
                double x = m3[1, 1];
            }

            Console.ReadLine(); // Test 1: approx 500ms
        }
    }
}
