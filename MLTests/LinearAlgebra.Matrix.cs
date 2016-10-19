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

        #region Property Tests
        /// <summary>
        /// Test transposing a Matrix (i.e. columns become rows)
        /// </summary>
        [TestMethod]
        public void TransposeMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            Matrix m2 = m1.Transpose;

            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 4.0 },
                { 2.0, 5.0 },
                { 3.0, 6.0 }
            });

            Assert.AreEqual(expectedResult, m2);

        }

        [TestMethod]
        public void Inverse2x2Matrix()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            Matrix m2 = m1.Inverse;

            Matrix expectedResult = new Matrix(new double[,] {
                { -2.0, 1.0 },
                { 1.5, -0.5 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void Inverse3x3Matrix()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 0.0, 1.0 },
                { 2.0, 0.0, 4.0 },
                { 0.0, 1.0, 1.0 }
            });

            Matrix m2 = m1.Inverse;

            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0,-0.5, 0.0 },
                { 1.0,-0.5, 1.0 },
                {-1.0, 0.5, 0.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        [ExpectedException(typeof(NonInvertibleMatrixException), "Matrix is not invertible.")]
        public void NonInvertibleMatrix()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 0.0, 1.0 },
                { 2.0, 0.0, 4.0 },
                { 5.0, 0.0, 6.0 }
            });

            Matrix m2 = m1.Inverse;
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Non-square Matrix is not invertible.")]
        public void InvertingNonSquareMatrix()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 2.0, 0.0, 4.0 },
                { 5.0, 0.0, 6.0 }
            });

            Matrix m2 = m1.Inverse;
        }
        #endregion

        #region Creation methods
        /// <summary>
        /// Test creating a 1x1 identity Matrix
        /// </summary>
        [TestMethod]
        public void Create1x1IdentityMatrix()
        {
            Matrix m1 = Matrix.Identity(1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });

            Assert.AreEqual(expectedResult, m1);

        }

        /// <summary>
        /// Test creating a 2x2 identity Matrix (i.e. diagonal values are 1, rest is zero)
        /// </summary>
        [TestMethod]
        public void Create2x2IdentityMatrix()
        {
            Matrix m1 = Matrix.Identity(2);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 0.0 },
                { 0.0, 1.0 }
            });

            Assert.AreEqual(expectedResult, m1);

        }

        /// <summary>
        /// Test creating a 4x4 identity Matrix (i.e. diagonal values are 1, rest is zero)
        /// </summary>
        [TestMethod]
        public void Create4x4IdentityMatrix()
        {
            Matrix m1 = Matrix.Identity(4);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 0.0, 0.0, 0.0 },
                { 0.0, 1.0, 0.0, 0.0 },
                { 0.0, 0.0, 1.0, 0.0 },
                { 0.0, 0.0, 0.0, 1.0 }
            });

            Assert.AreEqual(expectedResult, m1);

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

            Matrix m3 = m1 + m2; // 15,10,21 // 4,16,6
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

        [TestMethod]
        public void MultiplyingMatrixAndVectorTogether()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 }
            });
            Matrix v1 = new Matrix(new double[,] {
                { 7.0 },
                { 6.0 },
                { 5.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 60.0 },
                { 49.0 }
            });

            Matrix m2 = m1 * v1;
            Assert.AreEqual(expectedResult, m2);
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

        #region Element Operations (scalar)
        /// <summary>
        /// Test adding 5 to each matrix element
        /// </summary>
        [TestMethod]
        public void ElementAddFiveToMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0 }
            });

            Matrix m2 = Matrix.ElementAdd(m1, 5);
            Assert.AreEqual(expectedResult, m2);
        }

        /// <summary>
        /// Test subtracting five from each matrix element
        /// </summary>
        [TestMethod]
        public void ElementSubtractFiveFromMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            Matrix m2 = Matrix.ElementSubtract(m1, 5);
            Assert.AreEqual(expectedResult, m2);
        }

        /// <summary>
        /// Test multiplying each matrix element by 3
        /// </summary>
        [TestMethod]
        public void ElementMultiplyMatrixByThree()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0, 6.0, 9.0 },
                { 12.0, 15.0, 18.0 }
            });

            Matrix m2 = Matrix.ElementMultiply(m1, 3);
            Assert.AreEqual(expectedResult, m2);
        }

        /// <summary>
        /// Test dividing each matrix element by 10
        /// </summary>
        [TestMethod]
        public void ElementDivideMatrixByTen()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 0.1, 0.2, 0.3 },
                { 0.4, 0.5, 0.6 }
            });

            Matrix m2 = Matrix.ElementDivide(m1, 10);
            Assert.AreEqual(expectedResult, m2);
        }

        /// <summary>
        /// Test raising the power of each element by 2
        /// </summary>
        [TestMethod]
        public void ElementSquareMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 4.0, 9.0 },
                { 16.0, 25.0, 36.0 }
            });

            Matrix m2 = Matrix.ElementPower(m1, 2);
            Assert.AreEqual(expectedResult, m2);
        }

        /// <summary>
        /// Test getting the square root of each matrix element.
        /// </summary>
        [TestMethod]
        public void ElementSqrtMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 4.0, 9.0, 16.0 },
                { 25.0, 36.0, 49.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0 }
            });

            Matrix m2 = Matrix.ElementSqrt(m1);
            Assert.AreEqual(expectedResult, m2);
        }

        /// <summary>
        /// Test getting the absolute value of each matrix element.
        /// </summary>
        [TestMethod]
        public void ElementAbsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, -2.0, 3.0 },
                { -4.0, 5.0, -6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            Matrix m2 = Matrix.ElementAbs(m1);
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Element Operations (matrix)
        /// <summary>
        /// Test adding two Matrix objects together
        /// </summary>
        [TestMethod]
        public void ElementAddMatrixToMatrix()
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

            Matrix m3 = Matrix.ElementAdd(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test subtracting one Matrix from another
        /// </summary>
        [TestMethod]
        public void ElementSubtractMatrixFromMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 6.0, 6.0, 6.0 },
                { 6.0, 6.0, 6.0 }
            });

            Matrix m3 = Matrix.ElementSubtract(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test multiplying each matrix element together
        /// </summary>
        [TestMethod]
        public void ElementMultiplyMatrixByMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 1.0 },
                { 2.0, 1.0, 2.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 4.0, 3.0 },
                { 8.0, 5.0, 12.0 }
            });

            Matrix m3 = Matrix.ElementMultiply(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test dividing one Matrix by another
        /// </summary>
        [TestMethod]
        public void ElementDivideMatrixByMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 10.0, 2.0, 4.0 },
                { 6.0, 8.0, 2.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 1.0 },
                { 2.0, 1.0, 2.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 10.0, 1.0, 4.0 },
                { 3.0, 8.0, 1.0 }
            });

            Matrix m3 = Matrix.ElementDivide(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test that a null Matrix will throw a NullReferenceException exception.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(NullReferenceException), "Cannot add a null Matrix.")]
        public void ElementMultiplyMatrixByNullMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m2 = null;
            Matrix m3 = m1 * m2;
        }

        /// <summary>
        /// Test that two different sized Matrix objects will throw an
        /// InvalidMatrixDimensionsException exception.
        /// </summary>
        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Matrix dimensions must match.")]
        public void ElementMultiplyMatrixByNonMatchingMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 7.0, 8.0 },
                { 10.0, 11.0 }
            });
            Matrix m3 = m1 * m2;
        }
        #endregion

        #region Dimension Operations
        [TestMethod]
        public void SumOfMatrixColumns()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 5.0, 7.0, 9.0 }
            });

            Matrix m2 = Matrix.Sum(m1);
            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void SumOfMatrixRows()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 6.0 },
                { 15.0 }
            });

            Matrix m2 = Matrix.Sum(m1, MatrixDimensions.Rows);
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion


        #endregion

        #region Other Methods
        [TestMethod]
        public void SwapTwoExistingRows()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 },
                { 5.0, 6.0 }
            });

            m1.SwapRows(0, 1);

            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0, 4.0 },
                { 1.0, 2.0 },
                { 5.0, 6.0 }
            });

            Assert.AreEqual(expectedResult, m1);
        }


        [TestMethod]
        [ExpectedException(typeof(IndexOutOfRangeException), "Rows do not exist.")]
        public void SwapTwoNonExistentRows()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 },
                { 5.0, 6.0 }
            });

            m1.SwapRows(9, 10);
        }

        #endregion
    }
}
