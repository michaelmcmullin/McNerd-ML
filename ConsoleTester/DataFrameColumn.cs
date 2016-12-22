﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using McNerd.MachineLearning.LinearAlgebra;

namespace ConsoleTester
{
    /// <summary>
    /// The type of DataFrameColumn data
    /// </summary>
    public enum DataFrameColumnType { Ignore, Empty, Double, Factors, Bins }

    class DataFrameColumn
    {
        DataFrame parent;
        DataFrameColumnType columnType = DataFrameColumnType.Ignore;
        List<string> trainingRows;
        List<string> testRows;
        List<string> factors;
        Bins bins;

        string header;
        int columnCount = 1;
        string missingElement = String.Empty;
        double missingElementValue = 0;
        bool isResult = false;
        bool refresh = true;
        bool updateFactors = true;

        #region Constructors
        public DataFrameColumn(DataFrame parent)
        {
            Parent = parent;
        }

        public DataFrameColumn(DataFrame parent, Matrix m, int columnIndex)
        {
            Parent = parent;
            // Check if this column is already in the parent DataFrame. If not, add it.
            if (!parent.Columns.Contains(this))
                parent.Columns.Add(this);

            for (int i = 0; i < m.Rows; i++)
            {
                AddTrainingRow(m[i, columnIndex].ToString());
            }
        }

        #endregion

        /// <summary>
        /// Indexer to retrieve the original string value for a given row.
        /// </summary>
        /// <param name="row">The row to get/set the value of.</param>
        /// <returns>The string value contained in the given row.</returns>
        public string this[int row]
        {
            get { return row < trainingRows.Count ? trainingRows[row] : String.Empty; }
            set { if (row < trainingRows.Count) trainingRows[row] = value; }
        }

        /// <summary>
        /// The DataFrame that this column belongs to.
        /// </summary>
        public DataFrame Parent
        {
            get { return parent; }
            set { parent = value; }
        }

        /// <summary>
        /// Convert a given row into an array of values that can be exported
        /// to a Matrix later.
        /// </summary>
        /// <param name="row">The index of the row to export.</param>
        /// <returns>An array of double values that can be inserted into a
        /// Matrix.</returns>
        public double[] ExportMatrixRow(int row)
        {
            double[] output = new double[ColumnCount];
            double elementValue = 0;

            if (row >= TrainingRowCount)
            {
                for (int i = 0; i < columnCount; i++)
                {
                    output[i] = EmptyValue;
                }
            }
            else
            {
                switch (columnType)
                {
                    case DataFrameColumnType.Double:
                        if (!double.TryParse(trainingRows[row], out elementValue))
                            elementValue = EmptyValue;
                        output[0] = elementValue;
                        break;
                    case DataFrameColumnType.Factors:
                        if (factors != null)
                        {
                            for (int i = 0; i < columnCount; i++)
                            {
                                if (trainingRows[row] == factors[i])
                                    output[i] = 1.0;
                                else
                                    output[i] = 0.0;
                            }
                        }
                        break;
                    case DataFrameColumnType.Bins:
                        if (bins != null)
                        {
                            int binIndex = -1;
                            string cmpValue = trainingRows[row] == String.Empty ? EmptyElement : trainingRows[row];
                            double binValue = 0;

                            if (Double.TryParse(trainingRows[row], out binValue))
                            {
                                binIndex = bins.binIndex(binValue);
                            }

                            for (int i = 0; i < columnCount; i++)
                            {
                                if (i == binIndex)
                                    output[i] = 1.0;
                                else
                                    output[i] = 0.0;
                            }
                        }
                        break;
                    default:
                        for (int i = 0; i < columnCount; i++)
                            output[i] = EmptyValue;
                        break;
                }
            }

            return output;
        }

        /// <summary>
        /// Get/set the header row value
        /// </summary>
        public string Header
        {
            get { return header; }
            set { header = value; }
        }

        /// <summary>
        /// Get an expanded list of headers.
        /// </summary>
        public List<string> GetHeaders()
        {
            List<string> headers = new List<string>();
            if (refresh)
                SetColumnCount();

            switch (columnType)
            {
                case DataFrameColumnType.Factors:
                    headers = factors;
                    break;
                case DataFrameColumnType.Bins:
                    headers = bins.BinLabels;
                    break;
                default:
                    headers.Add(Header);
                    break;
            }

            return headers;
        }

        /// <summary>
        /// Add a row to the end of the training data list.
        /// </summary>
        /// <param name="s">A string representation of the row to add.</param>
        public void AddTrainingRow(string s)
        {
            if (trainingRows == null) trainingRows = new List<string>();
            trainingRows.Add(s);
            refresh = true;
        }

        /// <summary>
        /// Add a row to the end of the test data list.
        /// </summary>
        /// <param name="s">A string representation of the row to add.</param>
        public void AddTestRow(string s)
        {
            if (testRows == null) testRows = new List<string>();
            testRows.Add(s);
            refresh = true;
        }

        /// <summary>
        /// Gets/sets the type of values held in this column
        /// </summary>
        public DataFrameColumnType ColumnType
        {
            get { return columnType; }
            set { columnType = value; refresh = true; }
        }

        /// <summary>
        /// Get the number of training rows in this column
        /// </summary>
        public int TrainingRowCount
        {
            get
            {
                return trainingRows == null ? 0 : trainingRows.Count;
            }
        }

        /// <summary>
        /// Get the number of test rows in this column
        /// </summary>
        public int TestRowCount
        {
            get
            {
                return testRows == null ? 0 : testRows.Count;
            }
        }

        /// <summary>
        /// Get the number of columns that this column will be ultimately
        /// broken up into.
        /// </summary>
        public int ColumnCount
        {
            get
            {
                if (refresh)
                    SetColumnCount();
                return columnCount;
            }
        }

        /// <summary>
        /// Force a refresh of how many columns are required, and populate factored
        /// values if necessary.
        /// </summary>
        private void SetColumnCount()
        {
            columnCount = 1; // Default value

            switch (ColumnType)
            {
                case DataFrameColumnType.Factors:
                    if (trainingRows != null)
                    {
                        if (updateFactors || factors == null)
                            factors = trainingRows.Distinct().ToList();
                        columnCount = factors.Count;
                    }
                    break;
                case DataFrameColumnType.Bins:
                    if (bins != null)
                    {
                        columnCount = bins.BinCount;
                    }
                    break;
                default:
                    break;
            }
            refresh = false;
        }

        /// <summary>
        /// The default element to use if an element is missing.
        /// </summary>
        public string EmptyElement
        {
            get { return missingElement; }
            set { missingElement = value; }
        }

        /// <summary>
        /// The value to substitute when an element is missing.
        /// </summary>
        public double EmptyValue
        {
            get { return missingElementValue; }
            set { missingElementValue = value; missingElement = value.ToString(); }
        }

        /// <summary>
        /// Convert a subset of the column data to a string.
        /// </summary>
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.AppendLine($"Total Columns: {ColumnCount}");
            sb.AppendLine($"Total Rows: {TrainingRowCount}");
            sb.AppendLine($"Column Type: {ColumnType}");

            int maxRows = Math.Min(10, TrainingRowCount);
            List<string> headers = GetHeaders();

            foreach(string header in headers)
            {
                sb.Append($"{header}\t");
            }
            sb.Append("\n");

            for (int i = 0; i < maxRows; i++)
            {
                switch (ColumnType)
                {
                    case DataFrameColumnType.Factors:
                        for (int j = 0; j < headers.Count; j++)
                        {
                            if (trainingRows[i] == headers[j])
                                sb.Append($"Y\t");
                            //sb.Append($"{rows[i]}\t");
                            else
                                sb.Append("-\t");
                        }
                        sb.Append("\n");
                        break;
                    case DataFrameColumnType.Bins:
                        int binIndex = -1;
                        if (bins != null)
                        {
                            double val = 0;
                            if (Double.TryParse(trainingRows[i], out val))
                            {
                                binIndex = bins.binIndex(val);
                            }
                        }

                        for (int j = 0; j < headers.Count; j++)
                        {
                            if (binIndex == j)
                                sb.Append($"Y\t");
                            else
                                sb.Append("-\t");
                        }
                        sb.Append("\n");
                        break;
                    default:
                        sb.AppendLine(trainingRows[i]);
                        break;
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Indicates whether this column represents the results.
        /// </summary>
        public bool IsResult { get { return isResult; } set { isResult = value; } }

        /// <summary>
        /// Specify a number of bin values to use for this column.
        /// </summary>
        /// <param name="values">An array of values to use as bin boundaries.</param>
        public void SetBins(double[] values)
        {
            bins = new Bins();
            foreach (double val in values)
            {
                bins.AddBin(val);
            }
        }

        /// <summary>
        /// Copy the factors from another DataFrameColumn to this one
        /// </summary>
        /// <param name="other">The DataFrameColumn to copy factors from.</param>
        public void CopyFactors(DataFrameColumn other)
        {
            updateFactors = false;
            this.factors = other.factors;
            refresh = true;
            SetColumnCount();
        }

        /// <summary>
        /// Copy the bins from another DataFrameColumn to this one.
        /// </summary>
        /// <param name="other">The DataFrameColumn to copy the bins from.</param>
        public void CopyBins(DataFrameColumn other)
        {
            this.bins = other.bins;
            refresh = true;
            SetColumnCount();
        }
    }
}
