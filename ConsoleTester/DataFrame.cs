using McNerd.MachineLearning.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleTester
{
    class DataFrame
    {
        IDataImporter importer;
        List<DataFrameColumn> columns = new List<DataFrameColumn>();
        bool hasResults = false;

        /// <summary>
        /// Constructor to set the data import implementation.
        /// </summary>
        /// <param name="di">The IDataImporter to use to fetch external data
        /// from a file to fill this DataFrame.</param>
        public DataFrame(IDataImporter di)
        {
            importer = di;
        }

        /// <summary>
        /// Load data from a given file into this DataFrame
        /// </summary>
        /// <param name="path">The path of the file to load.</param>
        /// <param name="hasHeader">Indicates whether or not the data has a
        /// header row.</param>
        public void Load(string path, bool hasHeader, bool hasResults)
        {
            this.hasResults = hasResults;
            importer.Load(path, hasHeader, hasResults, this);
        }

        /// <summary>
        /// Export a Matrix containing only the features of this DataFrame. Exclude
        /// results column, if any.
        /// </summary>
        /// <returns>A Matrix object populated with the input data feature set.</returns>
        public Matrix ExportFeatures()
        {
            int maxRows = MaxRows;
            if (maxRows == 0) maxRows = 1;

            int totalColumnCount = TotalColumns;
            int columnCount = columns.Count;
            if (hasResults && totalColumnCount > 1) { totalColumnCount--; columnCount--; }

            Matrix features = new Matrix(maxRows, totalColumnCount);

            for (int row = 0; row < maxRows; row++)
            {
                int featureColumn = 0;

                for (int column = 0; column < columnCount; column++)
                {
                    double[] columnValues = columns[column].ExportMatrixRow(row);
                    for (int i = 0; i < columns[column].ColumnCount; i++)
                    {
                        features[row, featureColumn++] = columnValues[i];
                    }
                }
            }

            return features;
        }

        /// <summary>
        /// Retrieve a Matrix containing the results column of this DataFrame, assumed to
        /// be the final column.
        /// </summary>
        /// <returns>A Matrix containing results of the input data.</returns>
        public Matrix ExportResults()
        {
            int maxRows = MaxRows;
            if (maxRows == 0) maxRows = 1;
            int column = columns.Count - 1;

            Matrix results = new Matrix(maxRows, 1);

            if (hasResults)
            {
                for (int row = 0; row < maxRows; row++)
                {
                    double[] columnValues = columns[column].ExportMatrixRow(row);
                    results[row, 1] = columnValues[0]; // Results column should only have one column. If there are any
                                                       // additional columns, ignore. Maybe throwing an exception might
                                                       // be more appropriate, but we'll leave that for now.
                }
            }

            return results;
        }

        /// <summary>
        /// Get the maximum number of rows available in any of the columns.
        /// </summary>
        public int MaxRows
        {
            get
            {
                int maxRows = 0;
                foreach(DataFrameColumn column in columns)
                {
                    if (column.RowCount > maxRows) maxRows = column.RowCount;
                }
                return maxRows;
            }
        }

        /// <summary>
        /// Get the total number of columns that will ultimately make up the
        /// final Matrix. Note that some columns may in fact represent a group
        /// of columns, so we can't rely on a simple columns.Count
        /// </summary>
        public int TotalColumns
        {
            get
            {
                if (columns == null) return 1;
                int totalCount = 0;
                foreach (DataFrameColumn column in columns)
                    totalCount += column.ColumnCount;
                return totalCount;
            }
        }

        /// <summary>
        /// A List of DataFrameColumn objects that make up this DataFrame
        /// </summary>
        public List<DataFrameColumn> Columns
        {
            get { return columns; }
        }

        /// <summary>
        /// A List of headers for each column, before expanding into factors, bins, etc.
        /// </summary>
        public List<string> Headers
        {
            get
            {
                List<string> headers = new List<string>();
                foreach (DataFrameColumn col in Columns)
                {
                    headers.Add(col.Header);
                }
                return headers;
            }
        }

        /// <summary>
        /// A List of expanded headers, after each column has been broken up into its
        /// constituent parts (e.g. factors)
        /// </summary>
        public List<string> ExpandedHeaders
        {
            get
            {
                List<string> headers = new List<string>();
                foreach(DataFrameColumn col in Columns)
                {
                    headers.AddRange(col.GetHeaders());
                }
                return headers;
            }
        }
    }
}
