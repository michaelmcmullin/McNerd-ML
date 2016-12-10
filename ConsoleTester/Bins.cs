using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleTester
{
    class Bins
    {
        private List<double> bins;

        /// <summary>
        /// Add a new bin boundary.
        /// </summary>
        /// <param name="number">The number to add to the list of bins.</param>
        public void AddBin(double number)
        {
            if (bins == null) bins = new List<double>();
            if (bins.Contains(number)) return;

            bins.Add(number);
            bins.Sort();
        }

        /// <summary>
        /// Get a list of labels for this set of Bins.
        /// </summary>
        public List<string> BinLabels
        {
            get
            {
                if (bins == null || bins.Count == 0) return null;
                List<string> labels = new List<string>();

                if (bins.Count == 1) labels.Add(bins[0].ToString());
                else
                {
                    double lowerBound = bins[0];
                    for (int i = 1; i < bins.Count; i++)
                    {
                        labels.Add($"{lowerBound}-{bins[i]}");
                        lowerBound = bins[i];
                    }
                }
                return labels;
            }
        }

        /// <summary>
        /// Get the number of Bins in this instance.
        /// </summary>
        public int BinCount
        {
            get
            {
                if (bins == null) return 0;
                if (bins.Count == 1) return 1;
                return bins.Count - 1;
            }
        }

        /// <summary>
        /// Get the index of which bin a particular number belongs to. It belongs to a bin
        /// if it is greater or equal to the lower bound, and less than the upper bound.
        /// </summary>
        /// <param name="number">Check which bin this number belongs to.</param>
        /// <returns>The zero-based index of the bin that contains this number, or -1.</returns>
        public int binIndex(double number)
        {
            if (bins == null) return -1;
            if (bins.Count == 1) return bins[0] == number ? 0 : -1;

            double lowerBound = bins[0];
            for (int i = 1; i < bins.Count; i++)
            {
                if (lowerBound <= number && bins[i] > number) return i-1;
                lowerBound = bins[i];
            }

            return -1;
        }
    }
}
