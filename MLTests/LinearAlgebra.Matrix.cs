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
                { 1.0, 2.0, 3.0 },
                { 0.0, 1.0, 4.0 },
                { 5.0, 6.0, 0.0 }
            });

            Matrix m2 = m1.Inverse;

            Matrix expectedResult = new Matrix(new double[,] {
                {-24.0, 18.0, 5.0 },
                { 20.0,-15.0,-4.0 },
                { -5.0,  4.0, 1.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void InverseTricky3x3Matrix()
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
        public void Inverse3x3Identity()
        {
            Matrix m1 = Matrix.Identity(3);

            Matrix m2 = m1.Inverse;

            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 0.0, 0.0 },
                { 0.0, 1.0, 0.0 },
                { 0.0, 0.0, 1.0 }
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

        /// <summary>
        /// Test creating a 4x4 Matrix filled with ones
        /// </summary>
        [TestMethod]
        public void Create3x4MatrixFilledWithOnes()
        {
            Matrix m1 = Matrix.Ones(3, 4);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 1.0, 1.0, 1.0 },
                { 1.0, 1.0, 1.0, 1.0 },
                { 1.0, 1.0, 1.0, 1.0 }
            });

            Assert.AreEqual(expectedResult, m1);
        }

        /// <summary>
        /// Test creating a 4x4 Matrix filled with ones
        /// </summary>
        [TestMethod]
        public void Create4x4MatrixFilledWithOnes()
        {
            Matrix m1 = Matrix.Ones(4);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 1.0, 1.0, 1.0 },
                { 1.0, 1.0, 1.0, 1.0 },
                { 1.0, 1.0, 1.0, 1.0 },
                { 1.0, 1.0, 1.0, 1.0 }
            });

            Assert.AreEqual(expectedResult, m1);
        }

        #region Magic Square
        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create1x1MagicSquare()
        {
            Matrix m1 = Matrix.Magic(1);
            Matrix expectedResult = Matrix.Identity(1);
            Assert.AreEqual(expectedResult, m1);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Cannot create a Magic Square of dimension 2.")]
        public void Create2x2MagicSquare()
        {
            Matrix m1 = Matrix.Magic(2);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create3x3MagicSquare()
        {
            Matrix m1 = Matrix.Magic(3);
            Matrix expectedResult = new Matrix(new double[,] {
                { 8.0, 1.0, 6.0 },
                { 3.0, 5.0, 7.0 },
                { 4.0, 9.0, 2.0 }
            });

            Assert.AreEqual(expectedResult, m1);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create4x4MagicSquare()
        {
            Matrix m1 = Matrix.Magic(4);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 15.0, 14.0, 4.0 },
                { 12.0, 6.0, 7.0, 9.0 },
                { 8.0, 10.0, 11.0, 5.0 },
                { 13.0, 3.0, 2.0, 16.0 }
            });

            Assert.AreEqual(expectedResult, m1);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create5x5MagicSquare()
        {
            Matrix m1 = Matrix.Magic(5);
            Matrix expectedResult = new Matrix(new double[,] {
                { 17.0, 24.0, 1.0, 8.0, 15.0 },
                { 23.0, 5.0, 7.0, 14.0, 16.0 },
                { 4.0, 6.0, 13.0, 20.0, 22.0 },
                { 10.0, 12.0, 19.0, 21.0, 3.0 },
                { 11.0, 18.0, 25.0, 2.0, 9.0 }
            });

            Assert.AreEqual(expectedResult, m1);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create6x6MagicSquare()
        {
            Matrix m1 = Matrix.Magic(6);
            Matrix expectedResult = new Matrix(new double[,] {
                { 35.0, 1.0, 6.0, 26.0, 19.0, 24.0 },
                { 3.0, 32.0, 7.0, 21.0, 23.0, 25.0 },
                { 31.0, 9.0, 2.0, 22.0, 27.0, 20.0 },
                { 8.0, 28.0, 33.0, 17.0, 10.0, 15.0 },
                { 30.0, 5.0, 34.0, 12.0, 14.0, 16.0 },
                { 4.0, 36.0, 29.0, 13.0, 18.0, 11.0 }
            });

            Assert.AreEqual(expectedResult, m1);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create8x8MagicSquare()
        {
            Matrix m1 = Matrix.Magic(8);
            Assert.IsTrue(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create9x9MagicSquare()
        {
            Matrix m1 = Matrix.Magic(9);
            Assert.IsTrue(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create10x10MagicSquare()
        {
            Matrix m1 = Matrix.Magic(10);
            Matrix expectedResult = new Matrix(new double[,] {
                { 92.0, 99.0,  1.0,  8.0, 15.0, 67.0, 74.0, 51.0, 58.0, 40.0 },
                { 98.0, 80.0,  7.0, 14.0, 16.0, 73.0, 55.0, 57.0, 64.0, 41.0 },
                {  4.0, 81.0, 88.0, 20.0, 22.0, 54.0, 56.0, 63.0, 70.0, 47.0 },
                { 85.0, 87.0, 19.0, 21.0,  3.0, 60.0, 62.0, 69.0, 71.0, 28.0 },
                { 86.0, 93.0, 25.0,  2.0,  9.0, 61.0, 68.0, 75.0, 52.0, 34.0 },
                { 17.0, 24.0, 76.0, 83.0, 90.0, 42.0, 49.0, 26.0, 33.0, 65.0 },
                { 23.0,  5.0, 82.0, 89.0, 91.0, 48.0, 30.0, 32.0, 39.0, 66.0 },
                { 79.0,  6.0, 13.0, 95.0, 97.0, 29.0, 31.0, 38.0, 45.0, 72.0 },
                { 10.0, 12.0, 94.0, 96.0, 78.0, 35.0, 37.0, 44.0, 46.0, 53.0 },
                { 11.0, 18.0,100.0, 77.0, 84.0, 36.0, 43.0, 50.0, 27.0, 59.0 }
            });

            Assert.AreEqual(expectedResult, m1);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create11x11MagicSquare()
        {
            Matrix m1 = Matrix.Magic(11);
            Assert.IsTrue(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void Create12x12MagicSquare()
        {
            Matrix m1 = Matrix.Magic(12);
            Assert.IsTrue(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void CheckIfNonSquareIsMagicSquare()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 15.0, 14.0, 4.0 },
                { 12.0, 6.0, 7.0, 9.0 },
                { 8.0, 10.0, 11.0, 5.0 }
            });

            Assert.IsFalse(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void CheckIf1x1MatrixIsMagicSquare()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0 }
            });

            Assert.IsTrue(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void CheckIf2x2MatrixIsMagicSquare()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            Assert.IsFalse(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void CheckIfRepeatingNumbersMatrixIsMagicSquare()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 15.0, 14.0, 1.0 },
                { 9.0, 6.0, 7.0, 9.0 },
                { 8.0, 10.0, 8.0, 5.0 },
                { 13.0, 0.0, 2.0, 16.0 }
            });

            Assert.IsFalse(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void CheckIfMagicSquareIsMagicSquare()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 15.0, 14.0, 4.0 },
                { 12.0, 6.0, 7.0, 9.0 },
                { 8.0, 10.0, 11.0, 5.0 },
                { 13.0, 3.0, 2.0, 16.0 }
            });

            Assert.IsTrue(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void CheckIfMagicRowsIsNotMagicSquare()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 1.0, 2.0, 3.0, 4.0 },
                { 1.0, 2.0, 3.0, 4.0 },
                { 1.0, 2.0, 3.0, 4.0 }
            });

            Assert.IsFalse(m1.IsMagic);
        }

        [TestCategory("Matrix: Magic Square"), TestMethod]
        public void CheckIfNonMagicSquareIsMagicSquare()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });

            Assert.IsFalse(m1.IsMagic);
        }
        #endregion
        #endregion

        #region Operator Tests
        #region Addition
        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void AddingScalarAndMatrixTogether()
        {
            double number = 4.0;
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 5.0, 6.0, 7.0 },
                { 8.0, 9.0, 10.0 }
            });

            Matrix m2 = number + m1;
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void AddingMatrixAndScalarTogether()
        {
            double number = 4.0;
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 5.0, 6.0, 7.0 },
                { 8.0, 9.0, 10.0 }
            });

            Matrix m2 = m1 + number;
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion

        #region Subtraction
        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void SubractingMatrixFromScalar()
        {
            double number = 10.0;
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 9.0, 8.0, 7.0 },
                { 6.0, 5.0, 4.0 }
            });

            Matrix m2 = number - m1;
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void SubractingScalarFromMatrix()
        {
            double number = 10.0;
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { -9.0, -8.0, -7.0 },
                { -6.0, -5.0, -4.0 }
            });

            Matrix m2 = m1 - number;
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion

        #region Unary Operators
        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void NegatingAMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, -2.0, 3.0 },
                { 4.0, 5.0, -6.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { -1.0, 2.0,-3.0 },
                { -4.0,-5.0, 6.0 }
            });

            Matrix m2 = -m1;
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Multiplication
        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        [ExpectedException(typeof(NullReferenceException), "Cannot multiply a null Matrix.")]
        public void ScalarMultiplicationOfNullMatrix()
        {
            Matrix m1 = null;
            double scalar = 2;
            Matrix m2 = scalar * m1;
        }

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        [TestMethod]
        public void MultiplyByTransposeValidMatrices()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m3 = Matrix.MultiplyByTranspose(m1, m2);

            Matrix expectedResult = new Matrix(new double[,] {
                { 12.0, 39.0 },
                { 15.0, 39.0 },
                { 43.0, 112.0 }
            });

            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        public void MultiplyByTransposeSameSquareMatrixTwoParameters()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = Matrix.MultiplyByTranspose(m1, m1);

            Matrix expectedResult = new Matrix(new double[,] {
                { 45.0, 27.0, 78.0 },
                { 27.0, 30.0, 64.0 },
                { 78.0, 64.0,181.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void MultiplyTransposeByValidMatrices()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0 },
                { 2.0, 5.0 },
                { 9.0, 8.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 },
                { 7.0, 8.0, 9.0 }
            });
            Matrix m3 = Matrix.MultiplyTransposeBy(m1, m2);

            Matrix expectedResult = new Matrix(new double[,] {
                { 77, 94, 111 },
                { 79, 95, 111 }
            });

            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        public void MultiplyByTransposeSameMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0 },
                { 2.0, 5.0 },
                { 9.0, 8.0 }
            });
            Matrix m2 = Matrix.MultiplyByTranspose(m1);

            Matrix expectedResult = new Matrix(new double[,] {
                { 45.0, 27.0, 78.0 },
                { 27.0, 29.0, 58.0 },
                { 78.0, 58.0,145.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void MultiplyTransposeBySameMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = Matrix.MultiplyTransposeBy(m1);

            Matrix expectedResult = new Matrix(new double[,] {
                {117.0, 90.0, 54.0 },
                { 90.0, 73.0, 48.0 },
                { 54.0, 48.0, 36.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void MultiplyTransposeBySameMatrixTwoParameters()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = Matrix.MultiplyTransposeBy(m1, m1);

            Matrix expectedResult = new Matrix(new double[,] {
                {117.0, 90.0, 54.0 },
                { 90.0, 73.0, 48.0 },
                { 54.0, 48.0, 36.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void MultiplyTransposeBySameMatrixTwoInstancesTwoParameters()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m3 = Matrix.MultiplyTransposeBy(m1, m2);

            Matrix expectedResult = new Matrix(new double[,] {
                {117.0, 90.0, 54.0 },
                { 90.0, 73.0, 48.0 },
                { 54.0, 48.0, 36.0 }
            });

            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        public void MultiplyTransposeBySameSquareMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = Matrix.MultiplyTransposeBy(m1);

            Matrix expectedResult = new Matrix(new double[,] {
                {121.0,100.0, 56.0 },
                {100.0, 98.0, 53.0 },
                { 56.0, 53.0, 37.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void MultiplyTransposeBySameSquareMatrixTwoParameters()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = Matrix.MultiplyTransposeBy(m1, m1);

            Matrix expectedResult = new Matrix(new double[,] {
                {121.0,100.0, 56.0 },
                {100.0, 98.0, 53.0 },
                { 56.0, 53.0, 37.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestMethod]
        public void MultiplyTransposeBySameSquareMatrixTwoInstancesTwoParameters()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m3 = Matrix.MultiplyTransposeBy(m1, m2);

            Matrix expectedResult = new Matrix(new double[,] {
                {121.0,100.0, 56.0 },
                {100.0, 98.0, 53.0 },
                { 56.0, 53.0, 37.0 }
            });

            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Cannot multiply matrices of these dimensions.")]
        public void MultiplyTransposeInvalidMatrices()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0, 3.0, 0.0 },
                { 2.0, 5.0, 1.0 },
                { 9.0, 8.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0 },
                { 4.0, 5.0 }
            });
            Matrix m3 = Matrix.MultiplyByTranspose(m1, m2);
        }
        #endregion

        #region Division
        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void DivideScalarByMatrix()
        {
            double number = 12.0;
            Matrix m1 = new Matrix(new double[,] {
                { 2.0, 3.0, 4.0 },
                { 6.0, 10.0, 12.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 6.0, 4.0, 3.0 },
                { 2.0, 1.2, 1.0 }
            });

            Matrix m2 = number / m1;
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void DivideMatrixByScalar()
        {
            double number = 4.0;
            Matrix m1 = new Matrix(new double[,] {
                { 8.0, 12.0, 16.0 },
                { 20.0, 24.0, 28.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0 }
            });

            Matrix m2 = m1 / number;
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion

        #region Equality
        [TestCategory("Matrix: Operator Overloads"), TestMethod]
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

        #region Comparisons
        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void ElementsInMatrixLessThanScalar()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = m1 < 3;
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 1.0, 0.0 },
                { 0.0, 0.0, 0.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void ElementsInMatrixLessThanOrEqualToScalar()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = m1 <= 3;
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 1.0, 1.0 },
                { 0.0, 0.0, 0.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void ElementsInMatrixGreaterThanScalar()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = m1 > 3;
            Matrix expectedResult = new Matrix(new double[,] {
                { 0.0, 0.0, 0.0 },
                { 1.0, 1.0, 1.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Operator Overloads"), TestMethod]
        public void ElementsInMatrixGreaterThanOrEqualToScalar()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = m1 >= 3;
            Matrix expectedResult = new Matrix(new double[,] {
                { 0.0, 0.0, 1.0 },
                { 1.0, 1.0, 1.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Element Operations (scalar)
        /// <summary>
        /// Test adding 5 to each matrix element
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        [TestCategory("Matrix: Element Operations"), TestMethod]
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

        /// <summary>
        /// Test getting the absolute value of each matrix element.
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementExpMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 0.0, 1.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, Math.E }
            });

            Matrix m2 = Matrix.ElementExp(m1);
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion

        #region Element Operations (matrix)
        /// <summary>
        /// Test adding two Matrix objects together
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        /// Test adding Matrix and Vector objects together
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementAddVectorToMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0, 4.0, 6.0 },
                { 5.0, 7.0, 9.0 }
            });

            Matrix m3 = Matrix.ElementAdd(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test adding Matrix and Vector objects together
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementAddRowVectorToMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0, 3.0, 4.0 },
                { 6.0, 7.0, 8.0 }
            });

            Matrix m3 = Matrix.ElementAdd(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }


        /// <summary>
        /// Test subtracting one Matrix from another
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        /// Test subtracting a vector from a Matrix
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementSubtractVectorFromMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 6.0, 6.0, 6.0 },
                { 9.0, 9.0, 9.0 }
            });

            Matrix m3 = Matrix.ElementSubtract(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test subtracting a vector from a Matrix
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementSubtractRowVectorFromMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 6.0, 7.0, 8.0 },
                { 8.0, 9.0, 10.0 }
            });

            Matrix m3 = Matrix.ElementSubtract(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test multiplying each matrix element together
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        /// Test multiplying a Matrix by a vector.
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementMultiplyMatrixByVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 1.0 },
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 4.0, 3.0 },
                { 4.0, 10.0, 6.0 }
            });

            Matrix m3 = Matrix.ElementMultiply(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test multiplying a Matrix by a vector.
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementMultiplyMatrixByRowVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 8.0, 10.0, 12.0 }
            });

            Matrix m3 = Matrix.ElementMultiply(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test dividing one Matrix by another
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        /// Test dividing a Matrix by a vector
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementDivideMatrixByVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 10.0, 2.0, 4.0 },
                { 6.0, 8.0, 2.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0, 2.0, 1.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 10.0, 1.0, 4.0 },
                { 6.0, 4.0, 2.0 }
            });

            Matrix m3 = Matrix.ElementDivide(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test dividing a Matrix by a vector
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
        public void ElementDivideMatrixByRowVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 10.0, 2.0, 4.0 },
                { 6.0, 8.0, 2.0 }
            });
            Matrix m2 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 }
            });
            Matrix expectedResult = new Matrix(new double[,] {
                { 10.0, 2.0, 4.0 },
                { 3.0, 4.0, 1.0 }
            });

            Matrix m3 = Matrix.ElementDivide(m1, m2);
            Assert.AreEqual(expectedResult, m3);
        }

        /// <summary>
        /// Test that a null Matrix will throw a NullReferenceException exception.
        /// </summary>
        [TestCategory("Matrix: Element Operations"), TestMethod]
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
        [TestCategory("Matrix: Element Operations"), TestMethod]
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

        #region Statistical Operations
        #region Mean
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMeanColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.Mean(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.5, 3.5, 4.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMeanRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.Mean(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 },
                { 5.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMeanColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix m2 = Matrix.Mean(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMeanRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 3.0 }
            });
            Matrix m2 = Matrix.Mean(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Mean Square
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMeanSquareColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.MeanSquare(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 8.5, 14.5, 22.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMeanSquareRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 4.0, 5.0, 6.0, 7.0 }
            });
            Matrix m2 = Matrix.MeanSquare(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 7.5 },
                { 31.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMeanSquareColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 }
            });
            Matrix m2 = Matrix.MeanSquare(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 7.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMeanSquareRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 3.0 },
                { 4.0 }
            });
            Matrix m2 = Matrix.MeanSquare(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 7.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Max
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMaxColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 }
            });
            Matrix m2 = Matrix.Max(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 4.0, 5.0, 6.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMaxRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.Max(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0 },
                { 6.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMaxColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix m2 = Matrix.Max(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMaxRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 3.0 }
            });
            Matrix m2 = Matrix.Max(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Min
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMinColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 }
            });
            Matrix m2 = Matrix.Min(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMinRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.Min(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 },
                { 4.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMinColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix m2 = Matrix.Min(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMinRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 3.0 }
            });
            Matrix m2 = Matrix.Min(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Range
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetRangeColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 }
            });
            Matrix m2 = Matrix.Range(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0, 3.0, 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetRangeRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.Range(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 },
                { 2.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetRangeColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix m2 = Matrix.Range(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetRangeRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 3.0 }
            });
            Matrix m2 = Matrix.Range(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Median
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMedianColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 }
            });
            Matrix m2 = Matrix.Median(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.5, 3.5, 4.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMedianRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.Median(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 },
                { 5.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMedianColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 2.0, 4.0, 3.0 }
            });
            Matrix m2 = Matrix.Median(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetMedianRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0 },
                { 2.0 },
                { 3.0 }
            });
            Matrix m2 = Matrix.Median(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Quartile 1
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile1ColumnsFourRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m2 = Matrix.Quartile1(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.5, 3.5, 4.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile1ColumnsSixRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0 },
                { 16.0, 17.0, 18.0 }
            });
            Matrix m2 = Matrix.Quartile1(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 4, 5, 6 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile1ColumnsOddRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 },
                { 14.0, 15.0, 16.0 }
            });
            Matrix m2 = Matrix.Quartile1(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.25, 4.25, 5.25 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile1SingleRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 }
            });
            Matrix m2 = Matrix.Quartile1(m1, MatrixDimensions.Auto);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile1TwoRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 }
            });
            Matrix m2 = Matrix.Quartile1(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion

        #region Quartile 3
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile3ColumnsFourRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m2 = Matrix.Quartile3(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 8.5, 9.5, 10.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile3ColumnsSixRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0 },
                { 16.0, 17.0, 18.0 }
            });
            Matrix m2 = Matrix.Quartile3(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 13, 14, 15 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile3ColumnsOddRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 },
                { 14.0, 15.0, 16.0 }
            });
            Matrix m2 = Matrix.Quartile3(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 11, 12, 13 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile3SingleRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 }
            });
            Matrix m2 = Matrix.Quartile3(m1, MatrixDimensions.Auto);
            Matrix expectedResult = new Matrix(new double[,] {
                { 4.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetQuartile3TwoRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 }
            });
            Matrix m2 = Matrix.Quartile3(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 4.0, 5.0, 6.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion

        #region Interquartile Range
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetIQRColumnsFourRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 }
            });
            Matrix m2 = Matrix.IQR(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 6.0, 6.0, 6.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetIQRColumnsSixRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0 },
                { 16.0, 17.0, 18.0 }
            });
            Matrix m2 = Matrix.IQR(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 9.0, 9.0, 9.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetIQRColumnsOddRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 },
                { 7.0, 8.0, 9.0 },
                { 10.0, 11.0, 12.0 },
                { 14.0, 15.0, 16.0 }
            });
            Matrix m2 = Matrix.IQR(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 6.0, 6.0, 6.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetIQRSingleRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 }
            });
            Matrix m2 = Matrix.IQR(m1, MatrixDimensions.Auto);
            Matrix expectedResult = new Matrix(new double[,] {
                { 4.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetIQRTwoRowsMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 4.0, 2.0, 6.0 }
            });
            Matrix m2 = Matrix.IQR(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0, 3.0, 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion

        #region Mode
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetModeColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 5.0, 3.0 },
                { 1.0, 2.0, 3.0 },
                { 4.0, 2.0, 6.0 }
            });
            Matrix m2 = Matrix.Mode(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetModeRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 3.0, 3.0 },
                { 4.0, 4.0, 6.0 }
            });
            Matrix m2 = Matrix.Mode(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0 },
                { 4.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetModeColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 2.0, 4.0, 2.0 }
            });
            Matrix m2 = Matrix.Mode(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetModeRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 6.0 },
                { 2.0 },
                { 2.0 },
                { 3.0 }
            });
            Matrix m2 = Matrix.Mode(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetModeWhereNoMultipleValues()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix m2 = Matrix.Mode(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetModeWhereSeveralPossibilities()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 2.0, 2.0, 1.0, 1.0, 3.0, 3.0 }
            });
            Matrix m2 = Matrix.Mode(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetModeWhereSingleElement()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 3.0 }
            });
            Matrix m2 = Matrix.Mode(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        #endregion

        #region Variance
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetVarianceColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.Variance(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 4.5, 4.5, 4.5 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetVarianceRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.Variance(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 },
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetVarianceColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix m2 = Matrix.Variance(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetVarianceRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 3.0 }
            });
            Matrix m2 = Matrix.Variance(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #region Standard Deviation
        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetStandardDeviationColumnsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 4.0 },
                { 2.0, 5.0 },
                { 3.0, 6.0 }
            });
            Matrix m2 = Matrix.StandardDeviation(m1, MatrixDimensions.Columns);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetStandardDeviationRowsRectangularMatrix()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            Matrix m2 = Matrix.StandardDeviation(m1, MatrixDimensions.Rows);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 },
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetStandardDeviationColumnsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 }
            });
            Matrix m2 = Matrix.StandardDeviation(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetStandardDeviationColumnsLongerVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 7.0, 9.0 }
            });
            Matrix m2 = Matrix.StandardDeviation(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 2.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Statistical Operations"), TestMethod]
        public void GetStandardDeviationRowsVector()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 3.0 }
            });
            Matrix m2 = Matrix.StandardDeviation(m1);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }
            });
            Assert.AreEqual(expectedResult, m2);
        }
        #endregion

        #endregion
        #endregion

        #region Other Methods
        #region Swap Rows
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

        #region Join
        [TestMethod]
        public void JoinTwoValidColumnMatrices()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 4.0, 5.0 },
                { 7.0, 8.0 }
            });

            Matrix m2 = new Matrix(new double[,]
            {
                { 3.0 },
                { 6.0 },
                { 9.0 }
            });

            Matrix m3 = Matrix.Join(m1, m2, MatrixDimensions.Columns);

            Matrix expectedResult = new Matrix(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 },
                { 7.0, 8.0, 9.0 }
            });

            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Cannot join two columns with differing row counts.")]
        public void JoinTwoInvalidColumnMatrices()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 7.0, 8.0 }
            });

            Matrix m2 = new Matrix(new double[,]
            {
                { 3.0 },
                { 6.0 },
                { 9.0 }
            });

            Matrix m3 = Matrix.Join(m1, m2, MatrixDimensions.Columns);
        }

        [TestMethod]
        public void JoinTwoValidRowMatrices()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            Matrix m2 = new Matrix(new double[,]
            {
                { 5.0, 6.0 },
                { 7.0, 8.0 },
                { 9.0, 10.0 }
            });

            Matrix m3 = Matrix.Join(m1, m2, MatrixDimensions.Rows);

            Matrix expectedResult = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 },
                { 5.0, 6.0 },
                { 7.0, 8.0 },
                { 9.0, 10.0 }
            });

            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Cannot join two rows with differing column counts.")]
        public void JoinTwoInvalidRowMatrices()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 7.0, 8.0 }
            });

            Matrix m2 = new Matrix(new double[,]
            {
                { 3.0 },
                { 6.0 },
                { 9.0 }
            });

            Matrix m3 = Matrix.Join(m1, m2, MatrixDimensions.Rows);
        }

        [TestMethod]
        public void JoinTwoValidAutoColumnMatrices()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 4.0, 5.0 },
                { 7.0, 8.0 }
            });

            Matrix m2 = new Matrix(new double[,]
            {
                { 3.0 },
                { 6.0 },
                { 9.0 }
            });

            Matrix m3 = Matrix.Join(m1, m2, MatrixDimensions.Auto);

            Matrix expectedResult = new Matrix(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 },
                { 7.0, 8.0, 9.0 }
            });

            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        public void JoinTwoValidAutoRowMatrices()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            Matrix m2 = new Matrix(new double[,]
            {
                { 5.0, 6.0 },
                { 7.0, 8.0 },
                { 9.0, 10.0 }
            });

            Matrix m3 = Matrix.Join(m1, m2, MatrixDimensions.Auto);

            Matrix expectedResult = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 },
                { 5.0, 6.0 },
                { 7.0, 8.0 },
                { 9.0, 10.0 }
            });

            Assert.AreEqual(expectedResult, m3);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixDimensionsException), "Cannot join two matrices with differing rows and columns.")]
        public void JoinTwoInvalidAutoMatrices()
        {
            Matrix m1 = new Matrix(new double[,]
            {
                { 1.0, 2.0 },
                { 7.0, 8.0 }
            });

            Matrix m2 = new Matrix(new double[,]
            {
                { 3.0 },
                { 6.0 },
                { 9.0 }
            });

            Matrix m3 = Matrix.Join(m1, m2, MatrixDimensions.Auto);
        }

        #endregion

        #region Extract row/column
        [TestCategory("Matrix: Extract Rows/Columns"), TestMethod]
        public void ExtractValidMatrixRow()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });
            Matrix m2 = m1.GetRow(2);
            Matrix expectedResult = new Matrix(new double[,] {
                { 9.0, 10.0, 11.0, 12.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Extract Rows/Columns"), TestMethod]
        public void ExtractValidMatrixColumn()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });
            Matrix m2 = m1.GetColumn(2);
            Matrix expectedResult = new Matrix(new double[,] {
                { 3.0 }, { 7.0 }, { 11.0 }, { 15.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Extract Rows/Columns"), TestMethod]
        public void ExtractFirstMatrixRow()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });
            Matrix m2 = m1.GetRow(0);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Extract Rows/Columns"), TestMethod]
        public void ExtractLastMatrixRow()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });
            Matrix m2 = m1.GetRow(3);
            Matrix expectedResult = new Matrix(new double[,] {
                { 13.0, 14.0, 15.0, 16.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Extract Rows/Columns"), TestMethod]
        public void ExtractFirstMatrixColumn()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });
            Matrix m2 = m1.GetColumn(0);
            Matrix expectedResult = new Matrix(new double[,] {
                { 1.0 }, { 5.0 }, { 9.0 }, { 13.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Extract Rows/Columns"), TestMethod]
        public void ExtractLastMatrixColumn()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });
            Matrix m2 = m1.GetColumn(3);
            Matrix expectedResult = new Matrix(new double[,] {
                { 4.0 }, { 8.0 }, { 12.0 }, { 16.0 }
            });

            Assert.AreEqual(expectedResult, m2);
        }

        [TestCategory("Matrix: Extract Rows/Columns"), TestMethod]
        [ExpectedException(typeof(IndexOutOfRangeException), "Requested row is out of range.")]
        public void ExtractInvalidMatrixRow()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });
            Matrix m2 = m1.GetRow(10);
        }

        [TestCategory("Matrix: Extract Rows/Columns"), TestMethod]
        [ExpectedException(typeof(IndexOutOfRangeException), "Requested column is out of range.")]
        public void ExtractInvalidMatrixColumn()
        {
            Matrix m1 = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 },
                { 9.0, 10.0, 11.0, 12.0 },
                { 13.0, 14.0, 15.0, 16.0 }
            });
            Matrix m2 = m1.GetColumn(10);
        }

        #endregion
        #endregion
    }
}
