using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace ConsoleTester
{
    /// <summary>
    /// A test class implementing IDataImporter to load dummy data.
    /// </summary>
    class DataImporterDummy : IDataImporter
    {
        bool hasHeaders = false;
        bool hasResults = false;
        int columnCount = 0;
        int rowCount = 0;

        public void Load(string path, bool hasHeaderRow, bool hasResults, DataFrame data)
        {
            hasHeaders = hasHeaderRow;
            this.hasResults = hasResults;
            columnCount = 5 + (hasResults ? 1 : 0);

            data.Columns.Clear();
            for (int i = 0; i < columnCount; i++)
            {
                data.Columns.Add(new DataFrameColumn());
                data.Columns[i].ColumnType = DataFrameColumnType.Double;
            }

            if (hasHeaders)
                LoadHeaders(data.Columns);

            while (LoadNextRow(data.Columns))
                rowCount++;
        }

        private bool LoadNextRow(List<DataFrameColumn> columns)
        {
            if (rowCount < 10)
            {
                for (int i = 0; i < columnCount; i++)
                {
                    columns[i].AddRow((i * rowCount).ToString());
                }
                return true;
            }
            return false;
        }

        private void LoadHeaders(List<DataFrameColumn> columns)
        {
            for (int i = 0; i < columnCount; i++)
            {
                columns[i].Header = $"Item {i}";
            }

            if (hasResults)
                columns[columnCount - 1].Header = "Results";

            return;
        }
    }
}
