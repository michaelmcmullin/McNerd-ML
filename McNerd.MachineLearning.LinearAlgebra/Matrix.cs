using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace McNerd.MachineLearning.LinearAlgebra
{
    /// <summary>
    /// Describe which dimension of a matrix to work with.
    /// </summary>
    public enum MatrixDimensions { Auto, Rows, Columns }

    public class Matrix
    {
        #region Delegates
        /// <summary>
        /// General purpose delegate for processing a number and giving
        /// a result.
        /// </summary>
        /// <param name="a">The number to process.</param>
        /// <returns>The result of performing an operation on the number.</returns>
        public delegate double ProcessNumber(double a);

        /// <summary>
        /// General purpose delegate for processing two numbers and giving
        /// a result.
        /// </summary>
        /// <param name="a">The first number to process.</param>
        /// <param name="b">The second number to process.</param>
        /// <returns>The result of performing an operation on both inputs.</returns>
        public delegate double ProcessNumbers(double a, double b);

        /// <summary>
        /// General purpose delegate for processing a Matrix and giving
        /// a result.
        /// </summary>
        /// <param name="a">The Matrix to process.</param>
        /// <returns>The result of performing an operation on the Matrix.</returns>
        public delegate double ProcessMatrix(Matrix a);

        #endregion

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

        /// <summary>
        /// Get the transposed version of this Matrix (swap rows and columns)
        /// </summary>
        public Matrix Transpose
        {
            get
            {
                Matrix t = new Matrix(Columns, Rows);
                for (int index = 0; index < data.Length; index++)
                {
                    int i = index / Rows;
                    int j = index % Rows;
                    t.data[index] = data[(Columns * j) + i];
                }
                return t;
            }
        }

        /// <summary>
        /// Calculate the inverse of this Matrix.
        /// </summary>
        public Matrix Inverse
        {
            get
            {
                if (!IsSquare)
                    throw new InvalidMatrixDimensionsException("Inverse requires a Matrix to be square.");
                Matrix MResult = Matrix.Identity(Rows);

                for (int diagonal=0; diagonal < Rows; diagonal++)
                {
                    double diagonalValue = this[diagonal, diagonal];

                    // Ensure the diagonal value is not zero by swapping another row if necessary.
                    if (diagonalValue == 0)
                    {
                        for (int i=0; i<Rows; i++)
                        {
                            if (i != diagonal && this[i,diagonal] != 0 && this[diagonal,i] != 0)
                            {
                                this.SwapRows(diagonal, i);
                                MResult.SwapRows(diagonal, i);
                                diagonalValue = this[diagonal, diagonal];
                                break;
                            }
                        }
                        if (diagonalValue == 0)
                            throw new NonInvertibleMatrixException("This Matrix is not invertible");
                    }

                    int lineValueIndex = diagonal;
                    int itemIndex = 0;
                    int diagonalIndex = diagonal * this.Columns;

                    for (int row=0; row < Rows; row++)
                    {
                        if (row != diagonal)
                        {
                            double lineValue = this.data[lineValueIndex];
                            for (int column = 0; column < Columns; column++)
                            {
                                int diagonalColumnIndex = diagonalIndex + column;
                                this.data[itemIndex] = (this.data[itemIndex] * diagonalValue) - (this.data[diagonalColumnIndex] * lineValue);
                                MResult.data[itemIndex] = (MResult.data[itemIndex] * diagonalValue) - (MResult.data[diagonalColumnIndex] * lineValue);
                                itemIndex++;
                            }
                        }
                        else
                        {
                            itemIndex += this.Columns;
                        }
                        lineValueIndex += this.Columns;
                    }
                }

                // By now all the rows should be filled in...
                int indexResult = 0;
                int indexThis = 0;

                for (int i=0; i<Rows; i++)
                {
                    double divisor = this.data[indexThis];
                    indexThis += this.Columns + 1;

                    for (int j=0; j<Columns; j++)
                    {
                        MResult.data[indexResult++] /= divisor;
                    }
                }

                return MResult;
            }
        }

        /// <summary>
        /// Determine if this Matrix is a Magic Square.
        /// </summary>
        public bool IsMagic
        {
            get
            {
                if (Columns != Rows) return false;
                if (Columns == 2) return false;

                double sum = 0;
                double diagonalSum = 0;
                double reverseDiagonalSum = 0;
                var usedElements = new Dictionary<double, double>();

                for (int i = 0; i < Rows; i++)
                {
                    double rowSum = 0;
                    double columnSum = 0;

                    for (int j = 0; j < Columns; j++)
                    {
                        double thisElement = this[i, j];

                        // If we're reading the first row, we just calculate what all the other
                        // rows and columns should add up to.
                        if (i == 0)
                        {
                            sum += thisElement;
                        }
                        rowSum += thisElement;
                        columnSum += this[j, i];

                        if (!usedElements.ContainsKey(thisElement))
                            usedElements[thisElement] = 1;
                        else
                            return false; // This Matrix does not contain distinct numbers
                    }
                    if (rowSum != sum || columnSum != sum) return false;

                    diagonalSum += this[i, i];
                    reverseDiagonalSum += this[i, Columns - i - 1];
                }

                return reverseDiagonalSum == sum && diagonalSum == sum;
            }
        }
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
                Parallel.For(0, m1.rows, i => MultiplyRow(i, m1, m2, ref output));
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
            Parallel.For(0, m.rows, i => MultiplyRow(i, m, scalar, ref output));
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
        /// Convert this Matrix to a string.
        /// </summary>
        /// <returns>A string representation of this Matrix.</returns>
        /// <remarks>All elements are rounded to two decimal places.</remarks>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            int index = 0;
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    sb.AppendFormat("{0:0.00} ", data[index++]);
                }
                sb.Append("\n");
            }
            return sb.ToString();
        }

        #region Private row/column operations
        /// <summary>
        /// Calculate a single row result of multiplying two matrices.
        /// </summary>
        /// <param name="row">The zero-indexed row to calculate.</param>
        /// <param name="m1">The first matrix to multiply.</param>
        /// <param name="m2">The second matrix to multiply.</param>
        /// <param name="output">The matrix to store the results in.</param>
        private static void MultiplyRow(int row, Matrix m1, Matrix m2, ref Matrix output)
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
        private static void MultiplyRow(int row, Matrix m, double scalar, ref Matrix output)
        {
            int m_index = row * m.columns;

            for (int i = m_index; i < m_index + output.Columns; i++)
            {
                output.data[i] = scalar * m.data[i];
            }
        }

        /// <summary>
        /// Calculate a single row result of multiplying two matrices.
        /// </summary>
        /// <param name="row">The zero-indexed row from m1 to calculate.</param>
        /// <param name="m1">The first matrix to multiply.</param>
        /// <param name="m2">The second matrix to multiply.</param>
        /// <param name="output">The matrix to store the results in.</param>
        private static void MultiplyByTransposedRow(int row, Matrix m1, Matrix m2, ref Matrix output)
        {
            int m1_index = row * m1.columns;
            int output_index = row * output.Columns;
            int m2_index = 0;

            for (int column = 0; column < output.Columns; column++)
            {
                double result = 0;

                for (int i = 0; i < m1.Columns; i++)
                {
                    result += m1.data[m1_index + i] * m2.data[m2_index++];
                }

                output.data[output_index++] = result;
            }
        }

        /// <summary>
        /// Calculate a single row result of multiplying row transposed into a column, by a corresponding
        /// row in a second Matrix.
        /// </summary>
        /// <param name="column">The zero-indexed row from m1 to transpose into a column.</param>
        /// <param name="m1">The first matrix to multiply.</param>
        /// <param name="m2">The second matrix to multiply.</param>
        /// <param name="output">The matrix to store the results in.</param>
        private static void MultiplyByTransposedColumn(int column, Matrix m1, Matrix m2, ref Matrix output)
        {
            int output_index = column * output.Columns;
            int m1_index, m2_index;

            for (int m2Col = 0; m2Col < m2.Columns; m2Col++)
            {
                double result = 0;
                m1_index = column;
                m2_index = m2Col;

                for (int m2Row=0; m2Row < m2.Rows; m2Row++)
                {
                    result += m1.data[m1_index] * m2.data[m2_index];
                    m1_index += m1.Columns;
                    m2_index += m2.Columns;
                }

                output.data[output_index++] = result;
            }
        }

        /// <summary>
        /// Calculate a single row result of multiplying one matrix with its transpose.
        /// </summary>
        /// <param name="row">The zero-indexed row from m1 to calculate.</param>
        /// <param name="column">The zero-indexed column from m2 to calculate.</param>
        /// <param name="m1">The matrix to multiply with its transpose.</param>
        /// <param name="output">The matrix to store the results in.</param>
        private static void MultiplyByTransposedRow(int row, Matrix m1, ref Matrix output)
        {
            int m1_index = row * m1.columns;
            int output_index = row * output.Columns;
            int m2_index = 0;

            for (int column = 0; column < output.Columns; column++)
            {
                double result = 0;

                for (int i = 0; i < m1.Columns; i++)
                {
                    result += m1.data[m1_index + i] * m1.data[m2_index++];
                }

                output.data[output_index++] = result;
            }
        }

        /// <summary>
        /// Calculate a single row result of multiplying row transposed into a column, by a corresponding
        /// row in the original Matrix.
        /// </summary>
        /// <param name="column">The zero-indexed row from m1 to transpose into a column.</param>
        /// <param name="m1">The matrix to transpose and multiply by the original.</param>
        /// <param name="output">The matrix to store the results in.</param>
        private static void MultiplyByTransposedColumn(int column, Matrix m1, ref Matrix output)
        {
            MultiplyByTransposedColumn(column, m1, m1, ref output);
        }

        #endregion

        /// <summary>
        /// Multiply one Matrix by the transpose of the other.
        /// </summary>
        /// <param name="m1">The first Matrix to multiply.</param>
        /// <param name="m2">The Matrix to transpose and multiply.</param>
        /// <returns>The result of multiplying m1 with m2.Transpose.</returns>
        public static Matrix MultiplyByTranspose(Matrix m1, Matrix m2)
        {
            if (m1.Columns == m2.Columns)
            {
                Matrix output = new Matrix(m1.Rows, m2.Rows);
                Parallel.For(0, m1.Rows, i => MultiplyByTransposedRow(i, m1, m2, ref output));
                return output;
            }
            else
            {
                throw new InvalidMatrixDimensionsException("Multiplication cannot be performed on matrices with these dimensions.");
            }
        }

        /// <summary>
        /// Multiply a Matrix by its transpose.
        /// </summary>
        /// <param name="m1">The Matrix to multiply by its transpose.</param>
        /// <returns>The result of multiplying m1 with its transpose.</returns>
        public static Matrix MultiplyByTranspose(Matrix m1)
        {
            Matrix output = new Matrix(m1.Rows, m1.Rows);
            Parallel.For(0, m1.Rows, i => MultiplyByTransposedRow(i, m1, ref output));
            return output;
        }

        /// <summary>
        /// Multiply the Transpose of one Matrix by another Matrix.
        /// </summary>
        /// <param name="m1">The Matrix to transpose and multiply.</param>
        /// <param name="m2">The Matrix to multiply the Transpose of m1 by.</param>
        /// <returns>The result of multiplying m1.Transpose with m2.</returns>
        public static Matrix MultiplyTransposeBy(Matrix m1, Matrix m2)
        {
            if (m1.Rows == m2.Rows)
            {
                Matrix output = new Matrix(m1.Columns, m2.Columns);
                Parallel.For(0, m1.Columns, i => MultiplyByTransposedColumn(i, m1, m2, ref output));
                return output;
            }
            else
            {
                throw new InvalidMatrixDimensionsException("Multiplication cannot be performed on matrices with these dimensions.");
            }
        }

        /// <summary>
        /// Multiply the Transpose of a Matrix by the original Matrix.
        /// </summary>
        /// <param name="m1">The Matrix to transpose, and multiply by itself.</param>
        /// <returns>The result of multiplying m1.Transpose with m1.</returns>
        public static Matrix MultiplyTransposeBy(Matrix m1)
        {
            Matrix output = new Matrix(m1.Columns, m1.Columns);
            Parallel.For(0, m1.Columns, i => MultiplyByTransposedColumn(i, m1, ref output));
            return output;
        }

        /// <summary>
        /// Swap two rows in this Matrix.
        /// </summary>
        /// <param name="row1">The first row to swap.</param>
        /// <param name="row2">The second row to swap.</param>
        public void SwapRows(int row1, int row2)
        {
            double[] tmp = new double[Columns];
            int indexRow1 = row1 * Columns;
            int indexRow2 = row2 * Columns;

            if (indexRow1 > data.Length || indexRow2 > data.Length)
                throw new IndexOutOfRangeException("SwapRow method called with non-existent rows.");

            for (int i=0; i<Columns; i++)
            {
                tmp[i] = data[indexRow1 + i];
                data[indexRow1 + i] = data[indexRow2 + i];
                data[indexRow2 + i] = tmp[i];
            }
        }

        /// <summary>
        /// Join two Matrix objects together, side by side, or one above another.
        /// </summary>
        /// <param name="m1">The first Matrix to join.</param>
        /// <param name="m2">The second Matrix to join.</param>
        /// <param name="dimension">The dimensions to join them on.</param>
        /// <returns>A new Matrix containing both the original Matrices joined together.</returns>
        public static Matrix Join(Matrix m1, Matrix m2, MatrixDimensions dimension = MatrixDimensions.Auto)
        {
            Matrix result = null;
            switch(dimension)
            {
                case MatrixDimensions.Columns:
                    if (m1.Rows != m2.Rows)
                        throw new InvalidMatrixDimensionsException("Matrices cannot be joined as they don't have the same number of rows.");
                    result = new Matrix(m1.Rows, m1.Columns + m2.Columns);
                    int index = 0, indexM1 = 0, indexM2 = 0;
                    for (int row = 0; row < m1.Rows; row++)
                    {
                        for (int column = 0; column < m1.Columns; column++)
                        {
                            result.data[index++] = m1.data[indexM1++];
                        }
                        for (int column = 0; column < m2.Columns; column++)
                        {
                            result.data[index++] = m2.data[indexM2++];
                        }
                    }
                    break;
                case MatrixDimensions.Rows:
                    if (m1.Columns != m2.Columns)
                        throw new InvalidMatrixDimensionsException("Matrices cannot be joined as they don't have the same number of columns.");
                    result = new Matrix(m1.Rows + m2.Rows, m1.Columns);
                    for (int i = 0; i < m1.data.Length; i++)
                    {
                        result.data[i] = m1.data[i];
                    }
                    for (int i = 0; i < m2.data.Length; i++)
                    {
                        result.data[i + m1.data.Length] = m2.data[i];
                    }
                    break;
                case MatrixDimensions.Auto:
                    if (m1.Rows == m2.Rows)
                        goto case MatrixDimensions.Columns;
                    else
                        goto case MatrixDimensions.Rows;
                default:
                    break;
            }
            return result;
        }

        /// <summary>
        /// Extract a row from this Matrix.
        /// </summary>
        /// <param name="row">The zero-index row to extract.</param>
        /// <returns>A row-vector form Matrix.</returns>
        public Matrix GetRow(int row)
        {
            if (row >= this.Rows)
                throw new IndexOutOfRangeException("The requested row is out of range");
            Matrix result = new Matrix(1, this.Columns);

            int index = row * this.Columns;
            for (int i = 0; i < this.Columns; i++)
            {
                result.data[i] = this.data[index + i];
            }

            return result;
        }

        /// <summary>
        /// Extract a column from this Matrix.
        /// </summary>
        /// <param name="column">The zero-index column to extract.</param>
        /// <returns>A column-vector form Matrix.</returns>
        public Matrix GetColumn(int column)
        {
            if (column >= this.Columns)
                throw new IndexOutOfRangeException("The requested column is out of range");
            Matrix result = new Matrix(this.Rows, 1);

            int index = column;
            for (int i = 0; i < this.Rows; i++)
            {
                result.data[i] = this.data[index];
                index += this.Columns;
            }

            return result;
        }

        /// <summary>
        /// Fills the Matrix with a given number.
        /// </summary>
        /// <param name="number">The number to assign to every element in the Matrix.</param>
        public void Fill(double number)
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = number;
        }

        #region Matrix creation methods
        /// <summary>
        /// Create an identity matrix
        /// </summary>
        /// <param name="dimensions">The number of rows and columns for this matrix.</param>
        /// <returns>A square matrix with zeros everywhere, except for the main diagonal which is filled with ones.</returns>
        public static Matrix Identity(int dimensions)
        {
            Matrix Midentity = new Matrix(dimensions, dimensions);
            int index = 0;
            while (index < Midentity.data.Length)
            {
                Midentity.data[index] = 1;
                index += dimensions + 1;
            }

            return Midentity;
        }

        /// <summary>
        /// Create a rows*columns size Matrix filled with 1's.
        /// </summary>
        /// <param name="rows">The number of rows to initialise the matrix with.</param>
        /// <param name="cols">The number of columns to initialise the matrix with.</param>
        /// <returns>A Matrix object filled with 1's.</returns>
        public static Matrix Ones(int rows, int columns)
        {
            Matrix result = new Matrix(rows, columns);
            result.Fill(1.0);
            return result;
        }

        /// <summary>
        /// Create a square Matrix filled with 1's.
        /// </summary>
        /// <param name="dimensions">The number of rows and columns to initialise the
        /// matrix with. There will be an equal number of rows and columns.</param>
        /// <returns>A square Matrix object filled with 1's.</returns>
        public static Matrix Ones(int dimension)
        {
            Matrix result = new Matrix(dimension);
            result.Fill(1.0);
            return result;
        }

        /// <summary>
        /// Create a Magic Square for odd-numbered dimensions greater than 1.
        /// </summary>
        /// <param name="dimension">The dimension to use to create the Magic Square.</param>
        /// <returns>A Magic Square of the required dimensions.</returns>
        private static Matrix MagicSquareOdd(int dimension)
        {
            if (dimension <= 1)
                throw new InvalidMatrixDimensionsException("Dimensions must be greater than or equal to one.");

            if (dimension % 2 != 1)
                throw new InvalidMatrixDimensionsException("Dimensions must be an odd number.");

            Matrix output = new Matrix(dimension, dimension);

            // Set the first value and initialize current position to
            // halfway across the first row.
            int startColumn = dimension >> 1;
            int startRow = 0;
            output[startRow, startColumn] = 1;

            // Keep moving up and to the right until all squares are filled
            int newRow, newColumn;

            for (int i = 2; i <= dimension * dimension; i++)
            {
                newRow = startRow - 1; newColumn = startColumn + 1;
                if (newRow < 0) newRow = dimension - 1;
                if (newColumn >= dimension) newColumn = 0;

                if (output[newRow, newColumn] > 0)
                {
                    while (output[startRow, startColumn] > 0)
                    {
                        startRow++;
                        if (startRow >= dimension) startRow = 0;
                    }
                }
                else
                {
                    startRow = newRow; startColumn = newColumn;
                }
                output[startRow, startColumn] = i;
            }
            return output;
        }

        /// <summary>
        /// Create a Magic Square, where the numbers of each row, column and diagonal
        /// add up to the same number.
        /// </summary>
        /// <param name="dimensions">The number of rows and columns to initialise the
        /// matrix with. There will be an equal number of rows and columns.</param>
        /// <returns>A Magic Square Matrix.</returns>
        public static Matrix Magic(int dimension)
        {
            // Handle special cases first
            if (dimension == 1)
                return Matrix.Identity(1);
            if (dimension == 2)
                throw new InvalidMatrixDimensionsException("A Magic Square cannot have a dimension of 2");

            // Handle odd-numbered dimensions first
            if (dimension % 2 == 1)
            {
                return (MagicSquareOdd(dimension));
            }

            // Handle 'doubly-even' dimensions (divisible by 4)
            if (dimension % 4 == 0)
            {
                Matrix doubleEven = new Matrix(dimension, dimension);

                double index = 1;
                for (int i = 0, r = dimension-1; i < dimension; i++, r--)
                {
                    for (int j = 0, c = dimension-1; j < dimension; j++, c--)
                    {
                        // Fill in the diagonals
                        if (i == j || (j + i + 1) == dimension)
                        {
                            doubleEven[i, j] = index;
                        }
                        else
                        {
                            // Otherwise, fill in diagonally opposite element
                            doubleEven[r, c] = index;
                        }
                        index++;
                    }
                }

                return doubleEven;
            }

            // Other even dimensions (divisible by 2, but not 4) using Strachey's method.
            int k = dimension;
            int n = (k - 2) / 4;
            int qDimension = k / 2;
            int qElementCount = qDimension * qDimension;
            Matrix A = MagicSquareOdd(qDimension);
            Matrix B = Matrix.ElementAdd(A, qElementCount);
            Matrix C = Matrix.ElementAdd(B, qElementCount);
            Matrix D = Matrix.ElementAdd(C, qElementCount);

            // Exchange first n columns in A with n columns in D
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < A.Rows; j++)
                {
                    double tmp = D[j, i];
                    D[j, i] = A[j, i];
                    A[j, i] = tmp;
                }
            }

            // Exchange right-most n-1 columns in C with B.
            for (int i = C.Columns-n+1; i < C.Columns; i++)
            {
                for (int j = 0; j < A.Rows; j++)
                {
                    double tmp = C[j, i];
                    C[j, i] = B[j, i];
                    B[j, i] = tmp;
                }
            }

            // Exchange Middle and Centre-Middle squares in D with A
            int middle = (qDimension / 2);
            double swap = D[middle, 0]; D[middle, 0] = A[middle, 0]; A[middle, 0] = swap;
            swap = D[middle, middle]; D[middle, middle] = A[middle, middle]; A[middle, middle] = swap;


            Matrix row1 = Matrix.Join(A, C, MatrixDimensions.Columns);
            Matrix row2 = Matrix.Join(D, B, MatrixDimensions.Columns);

            Matrix output = Matrix.Join(row1, row2, MatrixDimensions.Rows);


            return output;
        }
        #endregion

        #region Element Operations
        /// <summary>
        /// Run a given operation on every element of a matrix.
        /// </summary>
        /// <param name="m">The Matrix to operate on.</param>
        /// <param name="number">The value to use in each operation.</param>
        /// <param name="operation">The delegate method to operate with.</param>
        /// <returns>A new Matrix with the original elements operated on appropriately.</returns>
        public static Matrix ElementOperation(Matrix m, double number, ProcessNumbers operation)
        {
            Matrix result = new Matrix(m.Rows, m.Columns);
            for (int i = 0; i < result.data.Length; i++)
                result.data[i] = operation(m.data[i], number);

            return result;
        }

        /// <summary>
        /// Run a given operation on every corresponding element in two Matrix
        /// objects with the same dimensions.
        /// </summary>
        /// <param name="m1">The first Matrix to operate on.</param>
        /// <param name="m2">The second Matrix to operate on.</param>
        /// <param name="operation">The delegate method to operate with.</param>
        /// <returns>A new Matrix with each element from both input Matrix objects
        /// operated on appropriately.</returns>
        public static Matrix ElementOperation(Matrix m1, Matrix m2, ProcessNumbers operation)
        {
            if (m1 == null || m2 == null)
                throw new ArgumentNullException("ElementOperation cannot accept null Matrix objects");
            if (!m1.HasSameDimensions(m2))
                throw new InvalidMatrixDimensionsException("ElementOperation requires both Matrix objects to have the same dimensions");

            Matrix result = new LinearAlgebra.Matrix(m1.Rows, m1.Columns);

            for (int i = 0; i < result.data.Length; i++)
                result.data[i] = operation(m1.data[i], m2.data[i]);

            return result;
        }

        /// <summary>
        /// Run a given operation on every element of a matrix.
        /// </summary>
        /// <param name="m">The Matrix to operate on.</param>
        /// <param name="operation">The delegate method to operate with.</param>
        /// <returns>A new Matrix with the original elements operated on appropriately.</returns>
        public static Matrix ElementOperation(Matrix m, ProcessNumber operation)
        {
            Matrix result = new Matrix(m.Rows, m.Columns);
            for (int i = 0; i < result.data.Length; i++)
                result.data[i] = operation(m.data[i]);

            return result;
        }

        #region Specific implementations of ElementOperation (scalars)
        /// <summary>
        /// Add a fixed number to each element in a given Matrix.
        /// </summary>
        /// <param name="m">The Matrix to process.</param>
        /// <param name="number">The number to add to each Matrix element.</param>
        /// <returns>A new Matrix containing elements added to the given number.</returns>
        public static Matrix ElementAdd(Matrix m, double number)
        {
            return ElementOperation(m, number, (x, y) => x + y);
        }

        /// <summary>
        /// Subtract a fixed number from each element in a given Matrix.
        /// </summary>
        /// <param name="m">The Matrix to process.</param>
        /// <param name="number">The number to subract from each Matrix element.</param>
        /// <returns>A new Matrix containing elements subtracted by the given number.</returns>
        public static Matrix ElementSubtract(Matrix m, double number)
        {
            return ElementOperation(m, number, (x, y) => x - y);
        }

        /// <summary>
        /// Multiply each element in a given Matrix by a fixed number.
        /// </summary>
        /// <param name="m">The Matrix to process.</param>
        /// <param name="number">The number to multiply each Matrix element by.</param>
        /// <returns>A new Matrix containing elements multiplied by the given number.</returns>
        public static Matrix ElementMultiply(Matrix m, double number)
        {
            return ElementOperation(m, number, (x, y) => x * y);
        }

        /// <summary>
        /// Divide each element in a given Matrix by a fixed number.
        /// </summary>
        /// <param name="m">The Matrix to process.</param>
        /// <param name="number">The number to divide each Matrix element by.</param>
        /// <returns>A new Matrix containing elements divided by the given number.</returns>
        public static Matrix ElementDivide(Matrix m, double number)
        {
            return ElementOperation(m, number, (x, y) => x / y);
        }

        /// <summary>
        /// Raise each element in a given Matrix by an exponent.
        /// </summary>
        /// <param name="m">The Matrix to process.</param>
        /// <param name="exponent">The exponent to raise each Matrix element by.</param>
        /// <returns>A new Matrix containing elements raised to the power of the given exponent.</returns>
        public static Matrix ElementPower(Matrix m, double exponent)
        {
            return ElementOperation(m, exponent, (x, y) => Math.Pow(x, y));
        }

        /// <summary>
        /// Multiply each element in a given Matrix by a fixed number.
        /// </summary>
        /// <param name="m">The Matrix to process.</param>
        /// <param name="number">The number to multiply each Matrix element by.</param>
        /// <returns>A new Matrix containing elements multiplied by the given number.</returns>
        public static Matrix ElementSqrt(Matrix m)
        {
            return ElementOperation(m, (x) => Math.Sqrt(x));
        }

        /// <summary>
        /// Multiply each element in a given Matrix by a fixed number.
        /// </summary>
        /// <param name="m">The Matrix to process.</param>
        /// <param name="number">The number to multiply each Matrix element by.</param>
        /// <returns>A new Matrix containing elements multiplied by the given number.</returns>
        public static Matrix ElementAbs(Matrix m)
        {
            return ElementOperation(m, (x) => Math.Abs(x));
        }

        #endregion

        #region Specific implementations of ElementOperation (matrices)
        /// <summary>
        /// Add the corresponding elements in two Matrix objects with the same dimensions.
        /// </summary>
        /// <param name="m1">The first Matrix to process.</param>
        /// <param name="m2">The second Matrix to add values to the first.</param>
        /// <returns>A new Matrix containing elements added from both input Matrix objects.</returns>
        public static Matrix ElementAdd(Matrix m1, Matrix m2)
        {
            return ElementOperation(m1, m2, (x, y) => x + y);
        }

        /// <summary>
        /// Subtract the corresponding elements in two Matrix objects with the same dimensions.
        /// </summary>
        /// <param name="m1">The first Matrix to process.</param>
        /// <param name="m2">The second Matrix to subtract values from the first.</param>
        /// <returns>A new Matrix containing elements subtracted from both input Matrix objects.</returns>
        public static Matrix ElementSubtract(Matrix m1, Matrix m2)
        {
            return ElementOperation(m1, m2, (x, y) => x - y);
        }

        /// <summary>
        /// Multiply the corresponding elements in two Matrix objects with the same dimensions.
        /// </summary>
        /// <param name="m1">The first Matrix to process.</param>
        /// <param name="m2">The second Matrix to multiply values from the first.</param>
        /// <returns>A new Matrix containing elements multiplied from both input Matrix objects.</returns>
        public static Matrix ElementMultiply(Matrix m1, Matrix m2)
        {
            return ElementOperation(m1, m2, (x, y) => x * y);
        }

        /// <summary>
        /// Divide the corresponding elements in two Matrix objects with the same dimensions.
        /// </summary>
        /// <param name="m1">The first Matrix to process.</param>
        /// <param name="m2">The second Matrix to divide values from the first.</param>
        /// <returns>A new Matrix containing elements divided from both input Matrix objects.</returns>
        public static Matrix ElementDivide(Matrix m1, Matrix m2)
        {
            return ElementOperation(m1, m2, (x, y) => x / y);
        }
        #endregion
        #endregion

        #region Dimension Operations
        /// <summary>
        /// Run a given operation on all elements in a particular dimension to reduce that dimension
        /// to a single row or column.
        /// </summary>
        /// <param name="m">The matrix to operate on.</param>
        /// <param name="dimension">Indicate whether to operate on rows or columns.</param>
        /// <param name="operation">The delegate method to operate with.</param>
        /// <returns>A matrix populated with the results of performing the given operation.</returns>
        /// <remarks>If the current matrix is a row or column vector, then a 1*1 matrix
        /// will be returned, regardless of which dimension is chosen. If the dimension is
        /// set to 'Auto', then the first non-singleton dimension is chosen. If no singleton
        /// dimension exists, then columns are used as the default.</remarks>
        public static Matrix ReduceDimension(Matrix m, MatrixDimensions dimension, ProcessNumbers operation)
        {
            Matrix result = null;

            // Process calculations
            switch(dimension)
            {
                case MatrixDimensions.Auto:
                    // Inspired by Octave, 'Auto' will process the first non-singleton dimension.
                    if (m.Rows == 1 || m.Columns == 1)
                    {
                        result = new Matrix(1, 1);
                        for (int i = 0; i < m.data.Length; i++)
                            result.data[0] = operation(result.data[0], m.data[i]);
                        return result;
                    }
                    else
                    {
                        // No singleton case? Let's go with columns.
                        goto case MatrixDimensions.Columns; // goto?? Haven't used one in years, and it feels good!!!!
                    }
                case MatrixDimensions.Columns:
                    result = new Matrix(1, m.Columns);
                    for (int i = 0; i < m.data.Length; i += m.Columns)
                        for (int j = 0; j < m.Columns; j++)
                            result.data[j] = operation(result.data[j], m.data[i + j]);
                    break;
                case MatrixDimensions.Rows:
                    result = new Matrix(m.Rows, 1);
                    int index = 0;
                    for (int i = 0; i < m.Rows; i++)
                        for (int j = 0; j < m.Columns; j++)
                            result.data[i] = operation(result.data[i], m.data[index++]);
                    break;
                default:
                    break;
            }

            return result;
        }

        #region Specific implementations of ReduceDimension
        /// <summary>
        /// Sum all elements along a specified dimension.
        /// </summary>
        /// <param name="m">The Matrix whose elements need to be added together.</param>
        /// <param name="dimension">The dimension (row or column) to process.</param>
        /// <returns>A 1*n or n*1 Matrix containing the sum of each element along the
        /// processed dimension.</returns>
        public static Matrix Sum(Matrix m, MatrixDimensions dimension = MatrixDimensions.Auto)
        {
            return ReduceDimension(m, dimension, (x, y) => x + y);
        }
        #endregion

        /// <summary>
        /// Run a set of operations on all elements in a particular dimension to reduce that dimension
        /// to a single row, and then perform an aggregate operation to produce a statistical 
        /// </summary>
        /// <param name="m">The matrix to operate on.</param>
        /// <param name="dimension">Indicate whether to operate on rows or columns.</param>
        /// <param name="operation">The delegate method to operate with.</param>
        /// <remarks>If the current matrix is a row or column vector, then a 1*1 matrix
        /// will be returned, regardless of which dimension is chosen. If the dimension is
        /// set to 'Auto', then the first non-singleton dimension is chosen. If no singleton
        /// dimension exists, then columns are used as the default.</remarks>
        public static Matrix StatisticalReduce(Matrix m, MatrixDimensions dimension, ProcessMatrix operation)
        {
            Matrix result = null;

            switch(dimension)
            {
                case MatrixDimensions.Auto:
                    if (m.Rows == 1)
                    {
                        result = new Matrix(1, 1);
                        result.data[0] = operation(m);
                        return result;
                    }
                    else if (m.Columns == 1)
                    {
                        result = new Matrix(1, 1);
                        result.data[0] = operation(m);
                        return result;
                    }
                    else
                    {
                        // No singleton case? Let's go with columns.
                        goto case MatrixDimensions.Columns;
                    }
                case MatrixDimensions.Columns:
                    result = new Matrix(1, m.Columns);
                    for (int i = 0; i < m.Columns; i++)
                        result.data[i] = operation(m.GetColumn(i));
                    break;
                case MatrixDimensions.Rows:
                    result = new Matrix(m.Rows, 1);
                    for (int i = 0; i < m.Rows; i++)
                        result.data[i] = operation(m.GetRow(i));
                    break;
                default:
                    break;
            }

            return result;
        }

        #region Specific implementations of StatisticalReduce
        /// <summary>
        /// Get the mean value of all elements in each dimension of a given Matrix.
        /// </summary>
        /// <param name="m">The Matrix whose elements need to be averaged.</param>
        /// <param name="dimension">The dimension (row or column) to process.</param>
        /// <returns>A 1*n or n*1 Matrix containing the mean of each element along the
        /// processed dimension.</returns>
        public static Matrix Mean(Matrix m, MatrixDimensions dimension = MatrixDimensions.Auto)
        {
            return StatisticalReduce(m, dimension, (x) => x.data.Average());
        }

        /// <summary>
        /// Get the maximum value of all elements in each dimension of a given Matrix.
        /// </summary>
        /// <param name="m">The Matrix to find the maximum value from.</param>
        /// <param name="dimension">The dimension (row or column) to process.</param>
        /// <returns>A 1*n or n*1 Matrix containing the maximum of each element along the
        /// processed dimension.</returns>
        public static Matrix Max(Matrix m, MatrixDimensions dimension = MatrixDimensions.Auto)
        {
            return StatisticalReduce(m, dimension, (x) => x.data.Max());
        }

        /// <summary>
        /// Get the minimum value of all elements in each dimension of a given Matrix.
        /// </summary>
        /// <param name="m">The Matrix to find the minimum value from.</param>
        /// <param name="dimension">The dimension (row or column) to process.</param>
        /// <returns>A 1*n or n*1 Matrix containing the minimum of each element along the
        /// processed dimension.</returns>
        public static Matrix Min(Matrix m, MatrixDimensions dimension = MatrixDimensions.Auto)
        {
            return StatisticalReduce(m, dimension, (x) => x.data.Min());
        }

        #endregion
        #endregion

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

    /// <summary>
    /// Custom excepction for matrix operations that require invertible matrices. 
    /// </summary>
    public class NonInvertibleMatrixException : InvalidOperationException
    {
        public NonInvertibleMatrixException()
        {
        }

        public NonInvertibleMatrixException(string message)
            : base(message)
        {
        }

        public NonInvertibleMatrixException(string message, Exception inner)
            : base(message, inner)
        {
        }
    }
}
