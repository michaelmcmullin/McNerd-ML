using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using McNerd.MachineLearning.LinearAlgebra;

namespace MLTests.LinearAlgebra
{
    [TestClass]
    public class MatrixTests
    {
        // Tests TODO:
        // Addition works as expected on valid matrices. That exception is thrown for uneven matrices.
        // What happens if a matrix is null? How are all the above tests effected?

        #region Constructor Tests
        /// <summary>
        /// Test the Matrix constructor using two ints to define dimensions.
        /// </summary>
        [TestMethod]
        public void ShouldCreateA10x12Matrix()
        {
            Matrix m1 = new Matrix(10, 12);
            Assert.AreEqual(10, m1.Dimensions[0]);
            Assert.AreEqual(12, m1.Dimensions[1]);
            Assert.AreEqual(10, m1.Rows);
            Assert.AreEqual(12, m1.Columns);
            Assert.IsFalse(m1.IsSquare);
        }

        /// <summary>
        /// Test the Matrix constructor using one int to define square dimensions.
        /// </summary>
        [TestMethod]
        public void ShouldCreateA20x20Matrix()
        {
            Matrix m1 = new Matrix(20);
            Assert.AreEqual(20, m1.Dimensions[0]);
            Assert.AreEqual(20, m1.Dimensions[1]);
            Assert.AreEqual(20, m1.Rows);
            Assert.AreEqual(20, m1.Columns);
            Assert.IsTrue(m1.IsSquare);
        }

        /// <summary>
        /// Test the Matrix constructor by passing a multidimensional array.
        /// </summary>
        [TestMethod]
        public void ShouldCreateA2x3Matrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Assert.AreEqual(2, m1.Dimensions[0]);
            Assert.AreEqual(3, m1.Dimensions[1]);
            Assert.AreEqual(2, m1.Rows);
            Assert.AreEqual(3, m1.Columns);
            Assert.IsFalse(m1.IsSquare);
        }
        #endregion

        #region Operator Tests
        [TestMethod]
        public void AddingTwoDefinedMatricesTogether()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 8.0, 10.0, 12.0 },
                { 14.0, 16.0, 18.0 }
            });

            Matrix m3 = m1 + m2;
            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        public void MultiplyingTwoDefinedMatricesTogether()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 7.0, 4.0 },
                { 6.0, 7.0 },
                { 5.0, 0.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 60.0, 45.0 },
                { 49.0, 43.0 },
                { 141.0, 92.0 }
            });

            Matrix m3 = m1 * m2;
            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        public void ScalarMultiplicationOfDefinedMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            double scalar = 2;
            Matrix expectedResult = new Matrix(new double[,] {
                { 12.0, 6.0, 0.0 },
                { 4.0, 10.0, 2.0 },
                { 18.0, 16.0, 12.0 }
            });

            Matrix m2 = scalar * m1;
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion
    }
}
