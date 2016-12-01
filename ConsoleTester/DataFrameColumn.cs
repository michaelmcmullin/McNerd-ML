using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleTester
{
    /// <summary>
    /// The type of DataFrameColumn data
    /// </summary>
    public enum DataFrameColumnType { Empty, Double, Factors }

    class DataFrameColumn
    {
        DataFrameColumnType columnType = DataFrameColumnType.Empty;
        List<string> rows;
        List<string> factors;
        string header;
        int columnCount = 1;
        string missingElement = String.Empty;
        double missingElementValue = 0;
        bool refresh = true;

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

            if (row >= RowCount)
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
                        if (!double.TryParse(rows[row], out elementValue))
                            elementValue = EmptyValue;
                        output[0] = elementValue;
                        break;
                    case DataFrameColumnType.Factors:
                        if (factors != null)
                        {
                            for (int i = 0; i < columnCount; i++)
                            {
                                if (rows[row] == factors[i])
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
                default:
                    headers.Add(Header);
                    break;
            }

            return headers;
        }

        /// <summary>
        /// Add a row to the end of the list.
        /// </summary>
        /// <param name="s">A string representation of the row to add.</param>
        public void AddRow(string s)
        {
            if (rows == null) rows = new List<string>();
            rows.Add(s);
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
        /// Get the number of rows in this column
        /// </summary>
        public int RowCount
        {
            get
            {
                return rows == null ? 0 : rows.Count;
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
                    if (rows != null)
                    {
                        factors = rows.Distinct().ToList();
                        columnCount = factors.Count;
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
            set { missingElementValue = value; }
        }
    }
}
