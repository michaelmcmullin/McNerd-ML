using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleTester
{
    /// <summary>
    /// A series of options that can be passed to a minimization function. Depending on the
    /// function, only a subset of properties will likely be used.
    /// </summary>
    class MinimizeOptions
    {
        public int InputLayerSize { get; set; }
        public int HiddenLayerSize { get; set; }
        public double[] Labels { get; set; }
    }
}
