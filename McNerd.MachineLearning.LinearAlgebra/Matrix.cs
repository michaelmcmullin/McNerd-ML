using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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
        double[,] data;

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
            data = new double[rows, columns];
        }

        /// <summary>
        /// Constructor to create a new square matrix.
        /// </summary>
        /// <param name="dimensions">The number of rows and columns to initialise the
        /// matrix with. There will be an equal number of rows and columns.</param>
        public Matrix(int dimensions) : this(dimensions, dimensions)
        {
        }

        public Matrix(double[,] array)
        {
            data = array;
            this.rows = array.GetLength(0);
            this.columns = array.GetLength(1);
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
            get { return data[row, column]; }
            set { data[row, column] = value; }
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
        /// <exception cref="System.ArgumentException">Thrown when both matrices have
        /// different dimensions.</exception>
        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            if (m1.HasSameDimensions(m2))
            {
                Matrix output = m1;
                for (int rows = 0; rows < m1.rows; rows++)
                {
                    for (int columns = 0; columns < m1.columns; columns++)
                    {
                        output[rows, columns] = m1[rows, columns] + m2[rows, columns];
                    }
                }
                return output;
            }
            else
            {
                throw new System.ArgumentException("Cannot add two Matrix objects whose dimensions do not match.");
            }
        }

        /// <summary>
        /// Override the == operator to compare matrix values.
        /// </summary>
        /// <param name="m1">The first matrix to compare.</param>
        /// <param name="m2">The second matrix to compare.</param>
        /// <returns>True if the values of both matrices match.</returns>
        public static bool operator ==(Matrix m1, Matrix m2)
        {
            if (null == m1 || null == m2) return false;
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
        #endregion
    }
}
