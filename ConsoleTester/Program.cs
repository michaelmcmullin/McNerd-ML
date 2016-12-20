using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using McNerd.MachineLearning.LinearAlgebra;

namespace ConsoleTester
{
    /// <summary>
    /// This console app is just for playing around with the code.
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            bool Exit = false;
            WriteCommands();

            while(!Exit)
            {
                Exit = GetCommand();
            }
        }

        static bool GetCommand()
        {
            bool IsExit = false;

            ConsoleKeyInfo key = Console.ReadKey(true);

            switch (key.KeyChar)
            {
                case '1':
                    Console.Clear();
                    LinearRegressionDemo();
                    WriteCommands();
                    break;
                case '2':
                    Console.Clear();
                    LogisticRegressionDemo();
                    WriteCommands();
                    break;
                case '3':
                    Console.Clear();
                    NeuralNetworkDemo();
                    WriteCommands();
                    break;
                case 't':
                    Console.Clear();
                    TitanicDemo();
                    WriteCommands();
                    break;
                case 'x':
                    IsExit = true;
                    break;
                default:
                    break;
            }

            return IsExit;
        }

        static void LinearRegressionDemo()
        {
            WriteH1("Linear Regression");

            #region Compute Cost
            WriteH2("Cost Functions");
            #region Test Cost Function A
            Matrix X = new Matrix(new double[,] {
                { 2.0, 1.0, 3.0 },
                { 7.0, 1.0, 9.0 },
                { 1.0, 8.0, 1.0 },
                { 3.0, 7.0, 4.0 }
            });
            Matrix y = new Matrix(new double[,] {
                { 2.0 },
                { 5.0 },
                { 5.0 },
                { 6.0 }
            });
            Matrix theta = new Matrix(new double[,] {
                { 0.4 },
                { 0.6 },
                { 0.8 }
            });

            double cost = LinearRegression.ComputeCost(X, y, theta);

            // Aiming for a result around 5.295
            Console.WriteLine("Target: 5.295    Actual: {0}", cost);
            #endregion

            #region Test Cost Function B
            X = new Matrix(new double[,] {
                { 1.0, 2.0 },
                { 1.0, 3.0 },
                { 1.0, 4.0 },
                { 1.0, 5.0 }
            });
            y = new Matrix(new double[,] {
                { 7.0 },
                { 6.0 },
                { 5.0 },
                { 4.0 }
            });
            theta = new Matrix(new double[,] {
                { 0.1 },
                { 0.2 }
            });

            cost = LinearRegression.ComputeCost(X, y, theta);

            // Aiming for a result around 11.945
            Console.WriteLine("Target: 11.945   Actual: {0}", cost);
            #endregion

            #region Test Cost Function C
            X = new Matrix(new double[,] {
                { 1.0, 2.0, 3.0 },
                { 1.0, 3.0, 4.0 },
                { 1.0, 4.0, 5.0 },
                { 1.0, 5.0, 6.0 }
            });
            y = new Matrix(new double[,] {
                { 7.0 },
                { 6.0 },
                { 5.0 },
                { 4.0 }
            });
            theta = new Matrix(new double[,] {
                { 0.1 },
                { 0.2 },
                { 0.3 }
            });

            cost = LinearRegression.ComputeCost(X, y, theta);

            // Aiming for a result around 7.0175
            Console.WriteLine("Target: 7.0175   Actual: {0}", cost);
            #endregion
            #endregion

            #region Gradient Descent
            WriteH2("Gradient Descent");

            #region Gradient Descent A
            X = new Matrix(new double[,] {
                { 2.0, 1.0, 3.0 },
                { 7.0, 1.0, 9.0 },
                { 1.0, 8.0, 1.0 },
                { 3.0, 7.0, 4.0 }
            });

            y = new Matrix(new double[,] {
                { 2.0 },
                { 5.0 },
                { 5.0 },
                { 6.0 }
            });

            theta = new Matrix(3, 1);
            Matrix result = LinearRegression.GradientDescent(X, y, theta, 0.01, 100);

            Console.Write("Target: 0.23; 0.56; 0.31;   Actual: ");
            Console.WriteLine(result.ToString().Replace(" \n", "; "));
            #endregion

            #region Gradient Descent B
            X = new Matrix(new double[,] {
                { 1.0, 5.0 },
                { 1.0, 2.0 },
                { 1.0, 4.0 },
                { 1.0, 5.0 }
            });

            y = new Matrix(new double[,] {
                { 1.0 },
                { 6.0 },
                { 4.0 },
                { 2.0 }
            });

            theta = new Matrix(2, 1);
            result = LinearRegression.GradientDescent(X, y, theta, 0.01, 1000);

            Console.Write("Target: 5.2; -0.57;   Actual: ");
            Console.WriteLine(result.ToString().Replace(" \n", "; "));
            #endregion

            #region Gradient Descent C
            // Starting from a non-zero theta
            X = new Matrix(new double[,] {
                { 1.0, 5.0 },
                { 1.0, 2.0 }
            });

            y = new Matrix(new double[,] {
                { 1.0 },
                { 6.0 }
            });

            theta = new Matrix(new double[,] { { 0.5 }, { 0.5 } });
            result = LinearRegression.GradientDescent(X, y, theta, 0.1, 10);

            Console.Write("Target: 1.7; 0.19;    Actual: ");
            Console.WriteLine(result.ToString().Replace(" \n", "; "));
            #endregion

            #endregion

            #region Feature Normalization
            WriteH2("Feature Normalization");

            #region Feature Normalization A
            X = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 3.0 }
            });
            result = LinearRegression.FeatureNormalization(X);

            Console.Write("Target: -1.0; 0.0; 1.0;    Actual: ");
            Console.WriteLine(result.ToString().Replace(" \n", "; "));
            #endregion

            #region Feature Normalization B
            X = Matrix.Magic(3);
            result = LinearRegression.FeatureNormalization(X);

            Console.Write("Target: 1.13 -1.00 0.38; -0.76 0.00 0.76; -0.38 1.00 -1.13;\nActual: ");
            Console.WriteLine(result.ToString().Replace(" \n", "; "));
            #endregion

            #region Feature Normalization C
            X = Matrix.Magic(3);
            X = Matrix.Join(Matrix.Ones(1, 3) * -1, X, MatrixDimensions.Rows);
            result = LinearRegression.FeatureNormalization(X);

            Console.Write("Target: -1.21 -1.01 -1.21; 1.21 -0.56 0.67; -0.14 0.34 0.95; 0.14 1.24 -0.41;\nActual: ");
            Console.WriteLine(result.ToString().Replace(" \n", "; "));
            #endregion

            #endregion

            #region Normal Equation
            WriteH2("Normal Equation");

            // This gives the same answer as Gradient Descent A above, if GD is allowed
            // to iterate enough times.
            X = new Matrix(new double[,] {
                { 2.0, 1.0, 3.0 },
                { 7.0, 1.0, 9.0 },
                { 1.0, 8.0, 1.0 },
                { 3.0, 7.0, 4.0 }
            });

            y = new Matrix(new double[,] {
                { 2.0 },
                { 5.0 },
                { 5.0 },
                { 6.0 }
            });

            theta = new Matrix(2, 1);
            Matrix thetaNormal = LinearRegression.NormalEquation(X, y);

            Console.Write("Target: 0.008; 0.568; 0.486;   Actual: ");
            Console.WriteLine(thetaNormal.ToString().Replace(" \n", "; "));
            #endregion
        }

        static void LogisticRegressionDemo()
        {
            WriteH1("Logistic Regression");

            #region Sigmoid Function
            WriteH2("Sigmoid Function");

            Matrix m1 = new Matrix(new double[,] { { 1200000 } });
            Matrix sigmoid1 = LogisticRegression.Sigmoid(m1);
            Console.Write("Target: 1.0   Actual: {0}", sigmoid1);

            m1[0, 0] = -25000;
            sigmoid1 = LogisticRegression.Sigmoid(m1);
            Console.Write("Target: 0.0   Actual: {0}", sigmoid1);

            m1[0, 0] = 0;
            sigmoid1 = LogisticRegression.Sigmoid(m1);
            Console.Write("Target: 0.5   Actual: {0}", sigmoid1);

            m1 = new Matrix(new double[,] { { 4, 5, 6 } });
            sigmoid1 = LogisticRegression.Sigmoid(m1);
            Console.Write("Target: 0.98 0.99 0.997   Actual: {0}", sigmoid1);
            #endregion

            #region Predict
            WriteH2("Prediction");

            m1 = new Matrix(new double[,] { { 1, 1 }, { 1, 2.5 }, { 1, 3 }, { 1, 4 } });
            Matrix theta = new Matrix(new double[,] { { -3.5 }, { 1.3 } });
            Matrix prediction = LogisticRegression.Predict(m1, theta);
            Console.WriteLine("Target: 0.0 ; 0.0 ; 1.0 ; 1.0 ;  Actual: {0}", prediction.ToString().Replace("\n", "; "));

            m1 = Matrix.Magic(3);
            theta = new Matrix(new double[,] { { 4 }, { 3 }, { -8 } });
            prediction = LogisticRegression.Predict(m1, theta);
            Console.WriteLine("Target: 0.0 ; 0.0 ; 1.0 ;        Actual: {0}", prediction.ToString().Replace("\n", "; "));
            #endregion

            #region Cost Function
            WriteH2("Cost Function");
            Matrix X = Matrix.AddIdentityColumn(Matrix.Magic(3));
            Matrix y = new Matrix(new double[,] { { 1 }, { 0 }, { 1 } });
            theta = new Matrix(new double[,] { { -2 }, { -1 }, { 1 }, { 2 } });
            Tuple<double, Matrix> cost = LogisticRegression.CostFunction(X, y, theta);

            Console.WriteLine("Target: 4.6832 ;  Actual: {0}", cost.Item1);

            #endregion

            #region Regularized Cost Function
            WriteH2("Regularized Cost Function");
            MinimizeOptions options = new MinimizeOptions();
            options.RegularizationParameter = 3;
            cost = LogisticRegression.CostFunction(X, y, theta, options);
            Console.WriteLine("Target: 7.6832 ;  Actual: {0}", cost.Item1);

            X = new Matrix(new double[,] {
                { 1.0, 0.1, 0.6, 1.1 },
                { 1.0, 0.2, 0.7, 1.2 },
                { 1.0, 0.3, 0.8, 1.3 },
                { 1.0, 0.4, 0.9, 1.4 },
                { 1.0, 0.5, 1.0, 1.5 }
            });
            y = new Matrix(new double[,] {
                { 1.0 },
                { 0.0 },
                { 1.0 },
                { 0.0 },
                { 1.0 }
            });
            theta = new Matrix(new double[,] { { -2 }, { -1 }, { 1 }, { 2 } });
            cost = LogisticRegression.CostFunction(X, y, theta, options);
            Console.WriteLine("Target: 2.5348 ;  Actual: {0}", cost.Item1);

            #endregion

            #region OneVsAll
            WriteH2("One vs All");
            X = new Matrix(new double[,] {
                { 8.0, 1.0, 6.0 },
                { 3.0, 5.0, 7.0 },
                { 4.0, 9.0, 2.0 },
                { 0.84147, 0.90930, 0.14112 },
                { 0.54030, -0.41615, -0.98999 }
            });
            y = new Matrix(new double[,] {
                { 1.0 },
                { 2.0 },
                { 2.0 },
                { 1.0 },
                { 3.0 }
            });
            //Matrix testTheta = new Matrix(4, 1);
            //Matrix X0 = Matrix.Join(Matrix.Ones(5, 1), X, MatrixDimensions.Columns);
            //cost = LogisticRegression.CostFunction(X0, y==1, testTheta, 0.1);
            //Console.WriteLine(cost.Item1);
            //Console.WriteLine(cost.Item2);

            double[] labels = new double[] { 1.0, 2.0, 3.0 };
            Matrix all_theta = LogisticRegression.OneVsAll(X, y, labels, 0.1);

            Console.WriteLine(all_theta);
            #endregion

            #region PredictOneVsAll
            WriteH2("Predict One vs All");
            X = new Matrix(new double[,] {
                { 1.0, 7.0 },
                { 4.0, 5.0 },
                { 7.0, 8.0 },
                { 1.0, 4.0 }
            });
            all_theta = new Matrix(new double[,] {
                { 1.0, -6.0, 3.0 },
                {-2.0,  4.0,-3.0 }
            });
            prediction = LogisticRegression.PredictOneVsAll(all_theta, X);
            Console.WriteLine("Target: 0; 1; 1; 0;    Actual: {0}", prediction.ToString().Replace("\n", "; "));
            #endregion
        }

        static void NeuralNetworkDemo()
        {
            WriteH1("Neural Network Regression");

            #region PredictNN
            WriteH2("Predict Neural Network");
            Matrix Theta1 = new Matrix(new double[,] {
                { 0.00000, 0.90930, -0.75680 },
                { 0.47943, 0.59847, -0.97753 },
                { 0.84147, 0.14112, -0.95892 },
                { 0.99749, -0.35078, -0.70554 }
            });
            Matrix Theta2 = new Matrix(new double[,] {
                { 0.00000, 0.93204, 0.67546, -0.44252, -0.99616 },
                { 0.29552, 0.99749, 0.42738, -0.68777, -0.92581 },
                { 0.56464, 0.97385, 0.14112, -0.87158, -0.77276 },
                { 0.78333, 0.86321, -0.15775, -0.97753, -0.55069 }
            });

            Matrix X = new Matrix(new double[,] {
                { 0.84147, 0.41212 },
                { 0.90930, -0.54402 },
                { 0.14112, -0.99999 },
                { -0.75680, -0.53657 },
                { -0.95892, 0.42017 },
                { -0.27942, 0.99061 },
                { 0.65699, 0.65029 },
                { 0.98936, -0.28790 }
            });
            Matrix prediction = NeuralNetwork.Predict(Theta1, Theta2, X);
            Console.WriteLine("Target: 3.00 ; 0.00 ; 0.00 ; 3.00 ; 3.00 ; 3.00 ; 3.00 ; 1.00 ;\nActual: {0}", prediction.ToString().Replace("\n", "; "));
            #endregion

            #region Sigmoid Gradient
            Matrix a = new Matrix ( new double[,] { { -1, -2, -3 } } );
            X = Matrix.Join(a, Matrix.Magic(3), MatrixDimensions.Rows);
            Matrix sg = NeuralNetwork.SigmoidGradient(X);

            WriteH2("Sigmoid Gradient");
            Console.WriteLine(sg);
            #endregion

            #region Neural Network Cost Function
            Matrix nn = new Matrix(new double[,] {
                { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8 }
            });
            int il = 2;
            int hl = 2;
            double[] labels = new double[] { 1, 2, 3, 4 };
            X = new Matrix(new double[,] {
                { 0.5403,  -0.41615 },
                {-0.98999, -0.65364 },
                { 0.28366,  0.96017 }
            });
            Matrix y = new Matrix(new double[,] {
                { 4 },
                { 2 },
                { 3 }
            });

            MinimizeOptions options = new MinimizeOptions();
            options.InputLayerSize = il;
            options.HiddenLayerSize = hl;
            options.Labels = labels;
            options.RegularizationParameter = 4;
            Tuple<double, Matrix> result = NeuralNetwork.NNCostFunction(X, y, nn, options);

            WriteH2("Neural Network Cost Function");
            Console.WriteLine($"J: {result.Item1} (Expected Result: 19.474)");
            Console.WriteLine(result.Item2);

            #endregion
        }

        static void TitanicDemo()
        {
            WriteH1("Testing");

            #region Dummy data importer
            //WriteH2("Dummy Data");
            //DataImporterDummy di = new DataImporterDummy();
            //DataFrame df = new DataFrame(di);
            //df.Load(String.Empty, true, true);

            //Console.WriteLine(df.TotalColumns);
            //foreach(string h in df.Headers)
            //{
            //    Console.WriteLine(h);
            //}
            #endregion

            #region CSV data importer
            WriteH2("CSV Data (Titanic)");
            DataImporterCSV di_csv = new DataImporterCSV();
            DataExporterCSV de_csv = new DataExporterCSV();

            DataFrame df_train = new DataFrame(di_csv, de_csv);
            DataFrame df_test = new DataFrame(di_csv, de_csv);

            df_train.Load(@"c:\temp\titanic.csv", true, "Survived");
            df_test.Load(@"c:\temp\titanic_test.csv", true);

            Console.WriteLine($"Total Columns (training data): {df_train.TotalColumns}");
            Console.WriteLine($"Total Columns (testing data):  {df_test.TotalColumns}");

            // Change the type of some of the training columns
            df_train.SetColumnType("pclass", DataFrameColumnType.Factors);
            df_train.SetColumnType("sex", DataFrameColumnType.Factors);
            df_train.SetColumnType("age", DataFrameColumnType.Bins);
            //df_train["age"].SetBins(new double[] { 0.0, 18.0, 100.0 });
            df_train["age"].SetBins(new double[] { 0.0, 15.0, 25.0, 30.0, 40.0, 50.0, 55.0, 65.0, 75.0, 100.0 });
            df_train["age"].EmptyValue = 30.27; // Average value of known ages
            df_train.SetColumnType("fare", DataFrameColumnType.Double);
            df_train.SetColumnType("sibsp", DataFrameColumnType.Double);
            df_train.SetColumnType("parch", DataFrameColumnType.Double);

            df_train.SetColumnType("survived", DataFrameColumnType.Double);

            // Try and match the types in the testing set
            df_test.MatchColumns(df_train);


            Console.WriteLine($"df_train hasResults? {df_train.HasResults}. df_test hasResults? {df_test.HasResults}");

            // Start calculations
            Matrix Xtrain = df_train.ExportFeatures();
            Matrix ytrain = df_train.ExportResults();

            Matrix Xtest = df_test.ExportFeatures();

            // Try Logistic Regression
            double[] labels = new double[] { 0.0, 1.0 };
            Matrix lr_theta = LogisticRegression.OneVsAll(Xtrain, ytrain, labels, 0.1, 1000);
            Matrix lr_prediction = LogisticRegression.PredictOneVsAll(lr_theta, Xtest);

            int input_layer_size = Xtrain.Columns;
            int output_layer_size = labels.Length;
            int hidden_layer_size = (input_layer_size + output_layer_size) / 2;
            Matrix[] nn_theta = NeuralNetwork.Train(Xtrain, ytrain, input_layer_size, hidden_layer_size, labels, 0.1, 1000);
            Matrix nn_prediction = NeuralNetwork.Predict(nn_theta[0], nn_theta[1], Xtest);

            // Exporting
            DataFrame df_lr_export = df_test;
            DataFrame df_nn_export = df_test;

            DataFrameColumn col_lr_results = new DataFrameColumn(df_lr_export, lr_prediction, 0);
            DataFrameColumn col_nn_results = new DataFrameColumn(df_nn_export, nn_prediction, 0);

            col_lr_results.Header = col_nn_results.Header = "Survived";

            df_lr_export.Save(@"c:\temp\lr_results.csv");
            df_nn_export.Save(@"c:\temp\nn_results.csv");
            #endregion
        }

        static void WriteCommands()
        {
            ConsoleColor fc = Console.ForegroundColor;
            ConsoleColor bc = Console.BackgroundColor;
            Console.ForegroundColor = ConsoleColor.Black;
            Console.BackgroundColor = ConsoleColor.Gray;

            Console.WriteLine("\n1:Linear Regression");
            Console.WriteLine("2:Logistic Regression");
            Console.WriteLine("3:Neural Networks");
            Console.WriteLine("t:Titanic [Kaggle.com]");
            Console.WriteLine("x:Exit");

            Console.ForegroundColor = fc;
            Console.BackgroundColor = bc;
        }

        static void WriteH1(string s)
        {
            ConsoleColor c = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(s.ToUpper());
            Console.ForegroundColor = c;
        }

        static void WriteH2(string s)
        {
            ConsoleColor c = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine(s);
            //Console.WriteLine(new String('-', 75));
            Console.ForegroundColor = c;
        }
    }
}
