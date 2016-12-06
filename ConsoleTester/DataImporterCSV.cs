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

        public void Load(string path, bool hasHeaderRow, DataFrame data)
        {
            using (TextFieldParser parser = new TextFieldParser(path))
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
                            data.Columns.Add(new DataFrameColumn());
                            if (hasHeaderRow)
                                data.Columns[i].Header = fields[i];
                            else
                                data.Columns[i].AddRow(fields[i]);
                        }
                        else
                        {
                            if (i < data.Columns.Count)
                                data.Columns[i].AddRow(fields[i]);
                        }
                        currentColumn = i;
                    }

                    // Check if a row has insufficient elements to fill.
                    if (currentColumn < (data.Columns.Count - 1))
                    {
                        for (int i = currentColumn; i < data.Columns.Count; i++)
                        {
                            data.Columns[i].AddRow(data.Columns[i].EmptyElement);
                        }
                    }

                    currentRow++;
                }
            }
        }
    }
}
