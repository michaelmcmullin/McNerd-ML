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
        int columnCount = 0;
        int rowCount = 0;

        public void Load(string pathToTrainingData, string pathToTestData, bool hasHeaderRow, DataFrame data)
        {
            hasHeaders = hasHeaderRow;
            columnCount = 5;

            data.Columns.Clear();
            for (int i = 0; i < columnCount; i++)
            {
                data.Columns.Add(new DataFrameColumn(data));
                data.Columns[i].ColumnType = DataFrameColumnType.Double;
            }

            if (hasHeaders)
                LoadHeaders(data.Columns);

            while (LoadNextRow(data.Columns))
                rowCount++;

            data.Columns[3].Header = "Test Header";
            data.Columns[2].ColumnType = DataFrameColumnType.Factors;
        }

        private bool LoadNextRow(List<DataFrameColumn> columns)
        {
            if (rowCount < 10)
            {
                for (int i = 0; i < columnCount; i++)
                {
                    columns[i].AddTrainingRow((i * rowCount).ToString());
                    columns[i].AddTestRow(((i * rowCount) + 1).ToString());
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

            return;
        }
    }
}
