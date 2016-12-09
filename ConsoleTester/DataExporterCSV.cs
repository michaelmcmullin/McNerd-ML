using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualBasic.FileIO;
using System.IO;

namespace ConsoleTester
{
    class DataExporterCSV : IDataExporter
    {
        string delimiter = ",";
        public string Delimiter { get { return delimiter; } set { delimiter = value; } }

        public void Save(string path, DataFrame df)
        {
            using (var file = new StreamWriter(path))
            {
                StringBuilder headers = new StringBuilder();

                for (int i = 0; i < df.Headers.Count; i++)
                {
                    headers.Append(CleanString(df.Headers[i]));
                    if (i < df.Headers.Count - 1)
                        headers.Append(delimiter);
                }
                file.WriteLine(headers);

                for (int row = 0; row < df.MaxRows; row++)
                {
                    StringBuilder sb = new StringBuilder();

                    for (int col = 0; col < df.Columns.Count; col++)
                    {
                        sb.Append(CleanString(df.Columns[col][row]));
                        if (col < df.Columns.Count - 1)
                            sb.Append(delimiter);
                    }

                    file.WriteLine(sb.ToString());
                    file.Flush();
                }
            }
        }

        /// <summary>
        /// Tidy up strings that contain characters that need escaping.
        /// </summary>
        /// <param name="s">The original string to examine.</param>
        /// <returns>A version of the string suitable for using in a CSV file.</returns>
        string CleanString(string s)
        {
            if (s == null)
                return String.Empty;

            if(s.Contains(delimiter) || s.Contains("\n") || s.Contains("\r"))
            {
                return $"\"{s.Replace("\"", "\"\"")}\"";
            }

            return s;
        }
    }
}
