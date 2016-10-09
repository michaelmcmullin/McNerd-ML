using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace McNerd.MachineLearning.LinearAlgebra
{
    /// <summary>
    /// Various special cases of matrix.
    /// </summary>
    public enum MatrixTypes { Zeros, Ones, Identity, Magic, Random }

    public class Matrix
    {
        #region Private Fields
        /// <summary>
        /// Storage array for the matrix data.
        /// </summary>
        double[] data;

        /// <summary>
        /// Dimensions of the matrix
        /// </summary>
        int rows, columns;
        #endregion

        #region Constructors
        /// <summary>
        /// Constructor to create a new matrix while specifying the number of
        /// rows and columns.
        /// </summary>
        /// <param name="rows">The number of rows to initialise the matrix with.</param>
        /// <param name="cols">The number of columns to initialise the matrix with.</param>
        public Matrix(int rows, int columns)
        {
            this.rows = rows;
            this.columns = columns;
            data = new double[rows * columns];
        }

        /// <summary>
        /// Constructor to create a new square matrix.
        /// </summary>
        /// <param name="dimensions">The number of rows and columns to initialise the
        /// matrix with. There will be an equal number of rows and columns.</param>
        public Matrix(int dimensions) : this(dimensions, dimensions)
        {
        }

        public Matrix(double[,] array) : this(array.GetLength(0), array.GetLength(1))
        {
            int index = 0;
            for (int row = 0; row < rows; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    data[index++] = array[row, column];
                }
            }
        }
        #endregion

        #region Indexers
        /// <summary>
        /// Indexer to easily access a specific location in this matrix.
        /// </summary>
        /// <param name="row">The row of the matrix location to access.</param>
        /// <param name="column">The column of the matrix location to access.</param>
        /// <returns>The value stored at the given row/column location.</returns>
        /// <remarks>Matrices are zero-indexed.</remarks>
        public double this[int row, int column]
        {
            get { return data[(row * Columns) + column]; }
            set { data[(row * Columns) + column] = value; }
        }
        #endregion

        #region Properties
        /// <summary>
        /// Indicates whether or not this matrix row and column dimensions are equal.
        /// </summary>
        public bool IsSquare => rows == columns;

        /// <summary>
        /// Get the dimensions of this matrix in a single-dimensional array of the form
        /// [rows,columns].
        /// </summary>
        public int[] Dimensions => new int[] { rows, columns };

        /// <summary>
        /// Get the number of rows in this matrix.
        /// </summary>
        public int Rows => rows;

        /// <summary>
        /// Get the number of columns in this matrix.
        /// </summary>
        public int Columns => columns;
        #endregion

        #region Operations
        /// <summary>
        /// Add two matrices together.
        /// </summary>
        /// <param name="m1">The first matrix to add.</param>
        /// <param name="m2">The second matrix to add.</param>
        /// <returns>The result of adding the two matrices together.</returns>
        /// <exception cref="InvalidMatrixDimensionsException">Thrown when both matrices have
        /// different dimensions.</exception>
        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            if (m1.HasSameDimensions(m2))
            {
                Matrix output = new Matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.data.Length; i++)
                {
                    output.data[i] = m1.data[i] + m2.data[i];
                }
                return output;
            }
            else
            {
                throw new InvalidMatrixDimensionsException("Cannot add two Matrix objects whose dimensions do not match.");
            }
        }

        /// <summary>
        /// Subtract one matrix from another.
        /// </summary>
        /// <param name="m1">The first matrix to subtract from.</param>
        /// <param name="m2">The second matrix to subtract from the first.</param>
        /// <returns>The result of subtracting the second matrix from the first.</returns>
        /// <exception cref="InvalidMatrixDimensionsException">Thrown when both matrices have
        /// different dimensions.</exception>
        public static Matrix operator -(Matrix m1, Matrix m2)
        {
            if (m1.HasSameDimensions(m2))
            {
                Matrix output = new Matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.data.Length; i++)
                {
                    output.data[i] = m1.data[i] - m2.data[i];
                }
                return output;
            }
            else
            {
                throw new InvalidMatrixDimensionsException("Cannot subtract two Matrix objects whose dimensions do not match.");
            }
        }

        /// <summary>
        /// Multiply two matrices together.
        /// </summary>
        /// <param name="m1">An nxm dimension matrix.</param>
        /// <param name="m2">An mxp dimension matrix.</param>
        /// <returns>An nxp Matrix that is the product of m1 and m2.</returns>
        /// <exception cref="InvalidMatrixDimensionsException">Thrown when the number of columns in the
        /// first matrix don't match the number of rows in the second matrix.</exception>
        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            if (m1.columns == m2.rows)
            {
                Matrix output = new Matrix(m1.rows, m2.columns);
                Parallel.For(0, m1.rows, i => MultiplyRow(i, m1, m2, output));
                return output;
            }
            else
            {
                throw new InvalidMatrixDimensionsException("Multiplication cannot be performed on matrices with these dimensions.");
            }
        }

        /// <summary>
        /// Scalar multiplication of a matrix.
        /// </summary>
        /// <param name="scalar">The scalar value to multiply each element of the matrix by.</param>
        /// <param name="m">The matrix to apply multiplication to.</param>
        /// <returns>A matrix representing the scalar multiplication of scalar * m.</returns>
        public static Matrix operator *(double scalar, Matrix m)
        {
            Matrix output = new Matrix(m.rows, m.columns);
            //for (int i = 0; i < m.data.Length; i++)
            //    output.data[i] = m.data[i] * scalar;
            //Parallel.For(0, m.data.Length, i => { output.data[i] = scalar * m.data[i]; });
            Parallel.For(0, m.rows, i => MultiplyRow(i, m, scalar, output));
            return output;
        }

        /// <summary>
        /// Scalar multiplication of a matrix.
        /// </summary>
        /// <param name="m">The matrix to apply multiplication to.</param>
        /// <param name="scalar">The scalar value to multiply each element of the matrix by.</param>
        /// <returns>A matrix representing the scalar multiplication of scalar * m.</returns>
        public static Matrix operator *(Matrix m, double scalar)
        {
            // Same as above, but ensuring commutativity - i.e. (s * m) == (m * s).
            return scalar * m;
        }

        /// <summary>
        /// Override the == operator to compare matrix values.
        /// </summary>
        /// <param name="m1">The first matrix to compare.</param>
        /// <param name="m2">The second matrix to compare.</param>
        /// <returns>True if the values of both matrices match.</returns>
        public static bool operator ==(Matrix m1, Matrix m2)
        {
            return m1.Equals(m2);
        }

        /// <summary>
        /// Override the != operator to compare matrix values.
        /// </summary>
        /// <param name="m1">The first matrix to compare.</param>
        /// <param name="m2">The second matrix to compare.</param>
        /// <returns>True if the values of both matrices differ.</returns>
        public static bool operator !=(Matrix m1, Matrix m2)
        {
            return !(m1 == m2);
        }
        #endregion

        #region Methods
        /// <summary>
        /// Indicates if this matrix has the same dimensions as another supplied matrix.
        /// </summary>
        /// <param name="other">Another matrix to compare this instance to.</param>
        /// <returns>true if both matrices have the same dimensions. Otherwise, false.</returns>
        public bool HasSameDimensions(Matrix other)
        {
            return (this.rows == other.rows) && (this.columns == other.columns);
        }

        /// <summary>
        /// Override the Object.Equals method to compare matrix values.
        /// </summary>
        /// <param name="obj">The object to compare to this matrix.</param>
        /// <returns>True if obj is a matrix, and its values match the current
        /// matrix values.</returns>
        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            Matrix m = obj as Matrix;
            if (object.ReferenceEquals(null, m)) return false;
            if (ReferenceEquals(this, m)) return true;

            if (!this.HasSameDimensions(m)) return false;

            for (int row = 0; row < rows; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    if (this[row, column] != m[row, column]) return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Compare this matrix with a second matrix by value.
        /// </summary>
        /// <param name="m">The matrix to compare to this one.</param>
        /// <returns>True if both matrices contain the same values.</returns>
        public bool Equals(Matrix m)
        {
            if (object.ReferenceEquals(null, m)) return false;
            if (ReferenceEquals(this, m)) return true;

            if (!this.HasSameDimensions(m)) return false;

            for (int row = 0; row < rows; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    if (this[row, column] != m[row, column]) return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Override the default hash code.
        /// </summary>
        /// <returns>A bitwise XOR based on rows and columns of this matrix.</returns>
        public override int GetHashCode()
        {
            return rows ^ columns;
        }

        /// <summary>
        /// Calculate a single row result of multiplying two matrices.
        /// </summary>
        /// <param name="row">The zero-indexed row to calculate.</param>
        /// <param name="m1">The first matrix to multiply.</param>
        /// <param name="m2">The second matrix to multiply.</param>
        /// <param name="output">The matrix to store the results in.</param>
        private static void MultiplyRow(int row, Matrix m1, Matrix m2, Matrix output)
        {
            int m1_index = row * m1.columns;
            int m2_index;

            for (int column = 0; column < output.Columns; column++)
            {
                double result = 0;
                m2_index = column;

                for (int i = 0; i < m1.Columns; i++)
                {
                    result += m1.data[m1_index + i] * m2.data[m2_index];
                    m2_index += m2.columns;
                }

                output[row, column] = result;

            }
        }

        /// <summary>
        /// Calculate the results of multiplying each element in a matrix
        /// row by a scalar value.
        /// </summary>
        /// <param name="row">The zero-indexed row to calculate.</param>
        /// <param name="m">The matrix to multiply by a scalar value.</param>
        /// <param name="scalar">The scalar value to multiply the matrix by.</param>
        /// <param name="output">The matrix that contains the results of multiplying the input
        /// matrix by a scalar value.</param>
        private static void MultiplyRow(int row, Matrix m, double scalar, Matrix output)
        {
            int m_index = row * m.columns;

            for (int i = m_index; i < m_index + output.Columns; i++)
            {
                output.data[i] = scalar * m.data[i];
            }
        }



        #endregion
    }

    /// <summary>
    /// Custom exception for matrix operations using incorrect dimensions.
    /// </summary>
    public class InvalidMatrixDimensionsException : InvalidOperationException
    {
        public InvalidMatrixDimensionsException()
        {
        }

        public InvalidMatrixDimensionsException(string message)
            : base(message)
        {
        }

        public InvalidMatrixDimensionsException(string message, Exception inner)
            : base(message, inner)
        {
        }
    }
}
