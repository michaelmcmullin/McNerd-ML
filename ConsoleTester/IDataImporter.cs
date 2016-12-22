using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleTester
{
    interface IDataImporter
    {
        void Load(string pathToTrainingData, string pathToTestData, bool hasHeaderRow, DataFrame data);
    }
}
