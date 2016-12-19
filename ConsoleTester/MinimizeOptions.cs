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
        /// <summary>
        /// For classification problems, this is an array of possible resulting classes.
        /// </summary>
        public double[] Labels { get; set; }

        /// <summary>
        /// The maximum number of iterations required.
        /// </summary>
        public int MaxIterations { get; set; }

        /// <summary>
        /// A parameter to help with regularization, to prevent overfitting the data.
        /// Leaving this at 0 turns off regularization.
        /// </summary>
        public double RegularizationParameter { get; set; }

        /// <summary>
        /// In Neural Networks, this represents the number of input features.
        /// </summary>
        public int InputLayerSize { get; set; }

        /// <summary>
        /// In Neural Networks, this represents the size of the hidden layer.
        /// </summary>
        public int HiddenLayerSize { get; set; }

    }
}
