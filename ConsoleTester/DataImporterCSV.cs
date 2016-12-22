using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualBasic.FileIO;

namespace ConsoleTester
{
    /// <summary>
    /// A DataImporter implementation for handling CSV files
    /// </summary>
    class DataImporterCSV : IDataImporter
    {
        string delimiter = ",";
        public string Delimiter { get { return delimiter; } set { delimiter = value; } }

        public void Load(string pathToTrainingData, string pathToTestData, bool hasHeaderRow, DataFrame data)
        {
            // Load Training Data
            #region Training Data
            using (TextFieldParser parser = new TextFieldParser(pathToTrainingData))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(Delimiter);
                int currentRow = 0;
                int currentColumn = 0;
                data.Columns.Clear();

                while (!parser.EndOfData)
                {
                    // Process row
                    string[] fields = parser.ReadFields();
                    for (int i = 0; i < fields.Length; i++)
                    {
                        if (currentRow == 0)
                        {
                            data.Columns.Add(new DataFrameColumn(data));
                            if (hasHeaderRow)
                                data.Columns[i].Header = fields[i];
                            else
                                data.Columns[i].AddTrainingRow(fields[i]);
                        }
                        else
                        {
                            if (i < data.Columns.Count)
                                data.Columns[i].AddTrainingRow(fields[i]);
                        }
                        currentColumn = i;
                    }

                    // Check if a row has insufficient elements to fill.
                    if (currentColumn < (data.Columns.Count - 1))
                    {
                        for (int i = currentColumn; i < data.Columns.Count; i++)
                        {
                            data.Columns[i].AddTrainingRow(data.Columns[i].EmptyElement);
                        }
                    }

                    currentRow++;
                }
            }
            #endregion

            // Load Test Data
            #region Test Data
            if (pathToTestData != String.Empty)
            {
                string[] headers = null;
                using (TextFieldParser parser = new TextFieldParser(pathToTestData))
                {
                    parser.TextFieldType = FieldType.Delimited;
                    parser.SetDelimiters(Delimiter);
                    int currentRow = 0;

                    while (!parser.EndOfData)
                    {
                        // Process row
                        string[] fields = parser.ReadFields();
                        for (int i = 0; i < fields.Length; i++)
                        {
                            if (currentRow == 0)
                            {
                                if (hasHeaderRow)
                                    headers = fields;
                            }
                            else
                            {
                                if (headers != null)
                                {
                                    DataFrameColumn col = data.FindColumn(headers[i]);
                                    if (col != null)
                                        col.AddTestRow(fields[i]);
                                }
                                else
                                {
                                    if (i < data.Columns.Count)
                                        data.Columns[i].AddTestRow(fields[i]);
                                }
                            }
                        }

                        // Check if a row has insufficient elements to fill.
                        for (int i = 0; i < data.Columns.Count; i++)
                        {
                            if (data.Columns[i].TestRowCount < data.MaxTestRows)
                                data.Columns[i].AddTestRow(data.Columns[i].EmptyElement);
                        }
                        currentRow++;
                    }
                }
            }
            #endregion

        }
    }
}
