using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleTester
{
    interface IDataImporter
    {
        void Load(string path, bool hasHeaderRow, bool hasResults, DataFrame data);
    }
}
