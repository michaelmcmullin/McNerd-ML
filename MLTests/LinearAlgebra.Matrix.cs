using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using McNerd.MachineLearning.LinearAlgebra;

namespace MLTests.LinearAlgebra
{
    [TestClass]
    public class MatrixTests
    {
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
        #region Addition
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
        [ExpectedException(typeof(NullReferenceException), "Cannot add a null Matrix.")]
        public void AddingDefinedMatrixToNullMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = null;

            Matrix m3 = m1 + m2;
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Matrix dimensions must match.")]
        public void AddingTwoUnevenDefinedMatricesTogether()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 7.0, 8.0 },
                { 10.0, 11.0 }
            });
            Matrix m3 = m1 + m2;
        }
        #endregion

        #region Subtraction
        [TestMethod]
        public void SubractingOneDefinedMatrixFromAnother()
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
                { -6.0, -6.0, -6.0 },
                { -6.0, -6.0, -6.0 }
            });

            Matrix m3 = m1 - m2;
            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        [ExpectedException(typeof(NullReferenceException), "Cannot subtract a null Matrix.")]
        public void SubtractingNullMatrixFromDefinedMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = null;

            Matrix m3 = m1 - m2;
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Matrix dimensions must match.")]
        public void SubractingTwoUnevenDefinedMatrices()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 7.0, 8.0 },
                { 10.0, 11.0 }
            });
            Matrix m3 = m1 - m2;
        }
        #endregion


        #region Multiplication
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
        [ExpectedException(typeof(NullReferenceException), "Cannot multiply a null Matrix.")]
        public void MultiplyingDefinedMatrixWithNull()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = null;

            Matrix m3 = m1 * m2;
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Matrix 1 column count must match matrix 2 row count.")]
        public void MultiplyingTwoIncorrectDimensionMatricesTogether()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 7.0, 4.0 },
                { 5.0, 0.0 }
            });

            Matrix m3 = m1 * m2;
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
            Matrix m3 = m1 * scalar;
            Assert.AreEqual(expectedResult, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        [ExpectedException(typeof(NullReferenceException), "Cannot multiply a null Matrix.")]
        public void ScalarMultiplicationOfNullMatrix()
        {
            Matrix m1 = null;
            double scalar = 2;
            Matrix m2 = scalar * m1;
        }

        #endregion

        #region Equality
        [TestMethod]
        public void EqualityOperatorOnTwoMatrices()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            Assert.IsTrue(m1 == m2);
        }
        #endregion
        #endregion
    }
}
