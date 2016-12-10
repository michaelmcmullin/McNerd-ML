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
        IDataExporter exporter;

        List<DataFrameColumn> columns = new List<DataFrameColumn>();
        bool hasResults = false;

        /// <summary>
        /// Constructor to set the data import implementation.
        /// </summary>
        /// <param name="di">The IDataImporter to use to fetch external data
        /// from a file to fill this DataFrame.</param>
        public DataFrame(IDataImporter di, IDataExporter de)
        {
            importer = di;
            exporter = de;
        }

        #region IO methods
        /// <summary>
        /// Load data from a given file into this DataFrame
        /// </summary>
        /// <param name="path">The path of the file to load.</param>
        /// <param name="hasHeader">Indicates whether or not the data has a
        /// header row.</param>
        /// <param name="resultColumn">The name of the column to set as the result
        /// column.</param>
        public void Load(string path, bool hasHeader, string resultColumn)
        {
            importer.Load(path, hasHeader, this);
            SetResultColumn(resultColumn);
        }

        /// <summary>
        /// Load data from a given file into this DataFrame
        /// </summary>
        /// <param name="path">The path of the file to load.</param>
        /// <param name="hasHeader">Indicates whether or not the data has a
        /// header row.</param>
        public void Load(string path, bool hasHeader)
        {
            importer.Load(path, hasHeader, this);
        }

        /// <summary>
        /// Save this DataFrame to a file.
        /// </summary>
        /// <param name="path">The path to save this DataFrame to.</param>
        public void Save(string path)
        {
            exporter.Save(path, this);
        }
        #endregion

        /// <summary>
        /// Use an Indexer to get a column.
        /// </summary>
        /// <param name="s">The case-insensitive name of the column to search for.</param>
        /// <returns>The first matching DataFrameColumn, or null.</returns>
        public DataFrameColumn this[string s]
        {
            get
            {
                return FindColumn(s);
            }
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

            int totalColumnCount = TotalActiveColumns;
            int columnCount = columns.Count;
            //if (hasResults && totalColumnCount > 1) { totalColumnCount--; columnCount--; }

            Matrix features = new Matrix(maxRows, totalColumnCount);

            for (int row = 0; row < maxRows; row++)
            {
                int featureColumn = 0;

                for (int column = 0; column < columnCount; column++)
                {
                    if (columns[column].ColumnType != DataFrameColumnType.Ignore && !columns[column].IsResult)
                    {
                        double[] columnValues = columns[column].ExportMatrixRow(row);
                        for (int i = 0; i < columns[column].ColumnCount; i++)
                        {
                            features[row, featureColumn++] = columnValues[i];
                        }
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
            int column = GetResultColumn();

            if (column < 0)
                return null;

            Matrix results = new Matrix(maxRows, 1);

            if (hasResults)
            {
                for (int row = 0; row < maxRows; row++)
                {
                    double[] columnValues = columns[column].ExportMatrixRow(row);
                    results[row, 0] = columnValues[0]; // Results column should only have one column. If there are any
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
        /// Get the total number of columns available in this DataFrame.
        /// Note that some columns may in fact represent a group
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
        /// Get the total number of active columns available in this DataFrame.
        /// If a column is set to a type of Ignore, then it is not included in
        /// this calculation. Result column is also ignored.
        /// </summary>
        public int TotalActiveColumns
        {
            get
            {
                if (columns == null) return 1;
                int totalCount = 0;
                foreach (DataFrameColumn column in columns)
                {
                    if (column.ColumnType != DataFrameColumnType.Ignore && !column.IsResult)
                        totalCount += column.ColumnCount;
                }
                return totalCount;
            }
        }

        /// <summary>
        /// Indicates whether this DataFrame has a result column.
        /// </summary>
        public bool HasResults { get { return hasResults; } set { hasResults = value; } }


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
                    if (col.ColumnCount == 1)
                        headers.AddRange(col.GetHeaders());
                    else
                    {
                        headers.AddRange(col.GetHeaders().Select(h => col.Header + ":" + h));
                    }
                }
                return headers;
            }
        }

        /// <summary>
        /// Find a column by its header name.
        /// </summary>
        /// <param name="header">The name of the header to search for.</param>
        /// <param name="caseSentitive">Indicates whether or not the search should
        /// be case sensitive.</param>
        /// <returns>The first matching DataFrameColumn, or null.</returns>
        public DataFrameColumn FindColumn(string header, bool caseSentitive = false)
        {
            header = caseSentitive ? header : header.ToLower();

            if (Columns != null)
            {
                foreach (DataFrameColumn column in Columns)
                {
                    string colHeader = caseSentitive ? column.Header : column.Header.ToLower();

                    if (colHeader == header)
                    {
                        return column;
                    }
                }
            }
            return null;
        }

        /// <summary>
        /// Given a DataFrame, df, ensure any column with the same name has matching
        /// attributes (i.e. ColumnType)
        /// </summary>
        /// <param name="df">The DataFrame whose columns dictate the attributes of
        /// this DataFrame.</param>
        public void MatchColumns(DataFrame df, bool caseSensitive = false)
        {
            if (Columns == null || Columns.Count == 0)
                return;

            foreach(DataFrameColumn column in Columns)
            {
                DataFrameColumn other = df.FindColumn(column.Header, caseSensitive);
                if (other != null)
                {
                    column.ColumnType = other.ColumnType;
                    column.CopyBins(other);
                }
            }
        }

        /// <summary>
        /// Set a column as the result column, ensuring that only one column can
        /// store results.
        /// </summary>
        /// <param name="header">The header name of the column that stores results.</param>
        /// <returns>true if the column was successfully set as a result column. Otherwise false.</returns>
        public bool SetResultColumn(string header)
        {
            DataFrameColumn column = FindColumn(header);
            if (column != null)
            {
                foreach (DataFrameColumn c in Columns)
                    c.IsResult = false;

                column.IsResult = hasResults = true;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Set a column as the result column, ensuring that only one column can
        /// store results.
        /// </summary>
        /// <param name="index">The 0-index of the column that stores results.</param>
        /// <returns>true if the column was successfully set as a result column. Otherwise false.</returns>
        public bool SetResultColumn(int index)
        {
            if (index < Columns.Count)
            {
                foreach (DataFrameColumn c in Columns)
                    c.IsResult = false;

                Columns[index].IsResult = hasResults = true;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Find the 0-index of the result columns.
        /// </summary>
        /// <returns>The zero based index of the column containing results. If none are
        /// found, return -1.</returns>
        public int GetResultColumn()
        {
            for (int i = 0; i < Columns.Count; i++)
                if (Columns[i].IsResult)
                    return i;

            return -1;
        }

        /// <summary>
        /// Set the type of a given column name.
        /// </summary>
        /// <param name="columnName">The name of the column to use.</param>
        /// <param name="columnType">The new type to define this column as.</param>
        /// <returns>true if the column is found, otherwise false.</returns>
        public bool SetColumnType(string columnName, DataFrameColumnType columnType)
        {
            DataFrameColumn column = FindColumn(columnName);
            if (column == null)
                return false;

            column.ColumnType = columnType;

            return true;
        }
    }
}
