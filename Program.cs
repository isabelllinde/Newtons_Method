using System;
using System.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Solvers;
using MathNet.Numerics.LinearAlgebra.Double.Solvers;
using MathNet.Numerics.LinearAlgebra.Factorization;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using MathNet.Numerics;
using MathNet.Numerics.Statistics;
using System.Diagnostics;

namespace FinalProject
{
    public class NewtonsMethod
    {
        public int[] t;
        public int kmax;
        public double e;
        public double[] y_t;

        public NewtonsMethod()
        {
            kmax = 10;
            y_t = new double[] { 51.1, 52.3, 54.2, 55.6, 56.1, 56.2, 56.4, 57.1, 57.9, 58.9, 60.3, 63.4, 65.8 };
            t = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
            e = 1e-8;
        }

        // Method calculating the Vector Function of Model1 1
        // Given a set of x's
        public double[] Function1(double[] x, int[] t, double[] y_t)
        {
            double[] F1 = new double[13];
            for (int i = 0; i < t.Length; i++)
            {
                F1[i] = y_t[i] - x[0] * Math.Exp(x[1] * t[i]);
            }
            return F1;
        }

        // Method calculating the Gradiant of f for Model 1
        //given a set of x's
        public double[] Gradient_f1(double[] x, int[] t, double[] y_t)
        {

            double df_dx1 = 0;
            double df_dx2 = 0;
            for (int i = 0; i < t.Length; i++)
            {
                df_dx1 += (-Math.Exp(x[1] * t[i]) * (y_t[i] - x[0] * Math.Exp(x[1] * t[i])));
                df_dx2 += (-t[i] * x[0] * Math.Exp(x[1] * t[i])) * (y_t[i] - x[0] * Math.Exp(x[1] * t[i]));
            }

            double[] grad1 = { df_dx1, df_dx2 };

            return grad1;
        }

        // Method calculating the Hessian matrix for Model 1
        public double[,] Hessian_f1(double[] x, int[] t, double[] y_t)
        {
            double df_dx1x1 = 0;
            double df_dx2x1 = 0;
            double df_dx1x2 = 0;
            double df_dx2x2 = 0;
            for (int i = 0; i < t.Length; i++)
            {
                df_dx1x1 += Math.Exp(2*x[1] + t[i]);
                df_dx2x1 += (t[i] * x[0] * Math.Exp(2 * t[i] * x[1])) - t[i] * Math.Exp(x[1] * t[i]) * (y_t[i] - x[0] * Math.Exp(x[1] * t[i]));
                df_dx1x2 += (t[i] * x[0] * Math.Exp(2 * t[i] * x[1])) - t[i] * Math.Exp(x[1] * t[i]) * (y_t[i] - x[0] * Math.Exp(x[1] * t[i]));
                df_dx2x2 += Math.Pow(t[i] * x[0], 2) * Math.Exp(2 * t[i] * x[1]) - Math.Pow(t[i], 2) * x[0] * Math.Exp(t[i] * x[1]) * (y_t[i] - x[0] * Math.Exp(t[i] * x[1]));
            }

            double[,] grad2_1 = { { df_dx1x1, df_dx1x2 }, { df_dx2x1, df_dx2x2 } };

            return grad2_1;
        }

        // Method for calculating the Function for Model 2
        // Outputs a vector
        public double[] Function2(double[] x, int[] t, double[] y_t)
        {
            double[] F2 = new double[t.Length];
            for (int i = 0; i < t.Length; i++)
            {
                F2[i] = (y_t[i] - (x[0] / (1 + x[1] * Math.Exp(x[2] * t[i]))));
            }
            return F2;
        }

        // Method for Calculating the Gradient of f for Model 2
        // Input is vector of x values, t and y arrays
        // Outputs a vector
        public double[] Gradient_f2(double[] x, int[] t, double[] y_t)
        {

            double df_dx1 = 0;
            double df_dx2 = 0;
            double df_dx3 = 0;

            for (int i = 0; i < t.Length; i++)
            {
                df_dx1 += -(y_t[i] * x[1] * Math.Exp(x[2] * t[i]) + y_t[i] - x[0]) / Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 2);
                df_dx2 += (x[0] * Math.Exp(x[2] * t[i]) * (y_t[i] * x[1] * Math.Exp(t[i] * x[2]) + y_t[i] - x[0])) / Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 3);
                df_dx3 += (t[i] * x[0] * x[1] * Math.Exp(x[2] * t[i]) * (y_t[i] * x[1] * Math.Exp(t[i] * x[2]) + y_t[i] - x[0])) / Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 3);
            }

            double[] grad2 = { df_dx1, df_dx2, df_dx3 };
            return grad2;
        }

        // Method to calculate the Hessian matrix for model 2
        // Input: vector of x values, t and y_t arrays
        // Output: Hessian matrix evaluated at x_k
        public double[,] Hessian_f2(double[] x, int[] t, double[] y_t)
        {
            double df_dx1x1 = 0;
            double df_dx2x1 = 0;
            double df_dx3x1 = 0;
            double df_dx1x2 = 0;
            double df_dx2x2 = 0;
            double df_dx3x2 = 0;
            double df_dx1x3 = 0;
            double df_dx2x3 = 0;
            double df_dx3x3 = 0;

            for (int i = 0; i < t.Length; i++)
            {
                df_dx1x1 += 1 / Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 2);
                df_dx2x2 += (-x[0] * Math.Exp(2 * x[2] * t[i]) * (2 * x[1] * y_t[i] * Math.Exp(x[2] * t[i]) - 3 * x[0] + y_t[i])) / (Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 4));
                df_dx3x3 += ((-t[i] * t[i] * x[0] * x[1] * Math.Exp(t[i] * x[2])) * (y_t[i] * x[1] * x[1] * Math.Exp(2 * t[i] * x[2]) - y_t[i] - 2 * x[0] * x[1] * Math.Exp(x[2] * t[i]) + x[0])) / (Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 4));
                df_dx2x1 += (Math.Exp(t[i] * x[2]) * (y_t[i] * (x[1] * Math.Exp(x[2] * t[i]) + y_t[i] - 2 * x[0]))) / (Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 3));
                df_dx1x2 += (Math.Exp(t[i] * x[2]) * (y_t[i] * (x[1] * Math.Exp(x[2] * t[i]) + y_t[i] - 2 * x[0]))) / (Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 3));
                df_dx3x1 += (x[1] * t[i] * Math.Exp(x[2] * t[i]) * (y_t[i] * x[1] * Math.Exp(t[i] * x[2]) + y_t[i] - x[0])) / (Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 3));
                df_dx1x3 += (x[1] * t[i] * Math.Exp(x[2] * t[i]) * (y_t[i] * x[1] * Math.Exp(x[2] * t[i]) + y_t[i] - x[0])) / (Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 3));
                df_dx2x3 += (-t[i] * x[0] * Math.Exp(x[2] * t[i]) * (y_t[i] * x[1] * x[1] * Math.Exp(2 * x[2] * t[i]) - x[0] * x[1] * Math.Exp(t[i] * x[2]) + x[0] - y_t[i])) / (Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 4));
                df_dx3x2 += (-t[i] * x[0] * Math.Exp(x[2] * t[i]) * (y_t[i] * x[1] * x[1] * Math.Exp(2 * x[2] * t[i]) - x[0] * x[1] * Math.Exp(t[i] * x[2]) + x[0] - y_t[i])) / (Math.Pow(x[1] * Math.Exp(x[2] * t[i]) + 1, 4));
            }

            double[,] grad2 = { { df_dx1x1, df_dx1x2, df_dx1x3 }, { df_dx2x1, df_dx2x2, df_dx2x3 }, { df_dx3x1, df_dx3x2, df_dx3x3 } };

            return grad2;
        }

        // Method to calculate the vector of Functions for model 3
        // Input: vector of x values, t and y_t array
        // Output: Vector of functions evaluated at x_k
        public double[] Function3(double[] x, int[] t, double[] y_t)
        {
            double[] F3 = new double[t.Length];
            for (int i = 0; i < t.Length; i++)
            {
                F3[i] = y_t[i] - x[0] - x[1] * t[i] - x[2] * Math.Pow(t[i], 2) - x[3] * Math.Pow(t[i], 3);
            }
            
            return F3;
        }

        // Method Calculating the Gradient of f for Model 3
        // Input: vector of x values, vector of t values, and vector of y_t values
        // Output: Gradient of f in vector format
        public double[] Gradient_f3(double[] x, int[] t, double[] y_t)
        {

            double df_dx1 = 0;
            double df_dx2 = 0;
            double df_dx3 = 0;
            double df_dx4 = 0;

            for (int i = 0; i < t.Length; i++)
            {
                df_dx1 += -y_t[i] + x[0] + x[1] * t[i] + x[2] * Math.Pow(t[i], 2) + x[3] * Math.Pow(t[1], 3);
                df_dx2 += -t[i] * (y_t[i] - x[0] - x[1] * t[i] - x[2] * Math.Pow(t[i], 2) - x[3] * Math.Pow(t[1], 3));
                df_dx3 += -Math.Pow(t[i], 2) * (y_t[i] - x[0] - x[1] * t[i] - x[2] * Math.Pow(t[i], 2) - x[3] * Math.Pow(t[1], 3));
                df_dx4 += -Math.Pow(t[i], 3) * (y_t[i] - x[0] - x[1] * t[i] - x[2] * Math.Pow(t[i], 2) - x[3] * Math.Pow(t[1], 3));
            }

            double[] grad3 = { df_dx1, df_dx2, df_dx3, df_dx4 };

            return grad3;
        }

        // Method to Calculate the Hessian of the functions for model 3
        // Input: vector of x values, vector of t values, and vector of y_t values
        // Output: Hessian matrix of f(x_k)
        public double[,] Hessian_f3(double[] x, int[] t, double[] y_t)
        {
            double df_dx1x1 = 0;
            double df_dx1x2 = 0;
            double df_dx1x3 = 0;
            double df_dx1x4 = 0;
            double df_dx2x1 = 0;
            double df_dx2x2 = 0;
            double df_dx2x3 = 0;
            double df_dx2x4 = 0;
            double df_dx3x1 = 0;
            double df_dx3x2 = 0;
            double df_dx3x3 = 0;
            double df_dx3x4 = 0;
            double df_dx4x1 = 0;
            double df_dx4x2 = 0;
            double df_dx4x3 = 0;
            double df_dx4x4 = 0;

            for (int i = 0; i < t.Length; i++)
            {
                df_dx1x1 += 1;
                df_dx1x2 += t[i];
                df_dx1x3 += Math.Pow(t[i], 2);
                df_dx1x4 += Math.Pow(t[i], 3);
                df_dx2x1 += t[i];
                df_dx2x2 += Math.Pow(t[i], 2);
                df_dx2x3 += Math.Pow(t[i], 3);
                df_dx2x4 += Math.Pow(t[i], 4);
                df_dx3x1 += Math.Pow(t[i], 2);
                df_dx3x2 += Math.Pow(t[i], 3);
                df_dx3x3 += Math.Pow(t[i], 4);
                df_dx3x4 += Math.Pow(t[i], 5);
                df_dx4x1 += Math.Pow(t[i], 3);
                df_dx4x2 += Math.Pow(t[i], 4);
                df_dx4x3 += Math.Pow(t[i], 5);
                df_dx4x4 += Math.Pow(t[i], 6);
            }

            double[,] grad3 = { { df_dx1x1, df_dx1x2, df_dx1x3, df_dx1x4 }, { df_dx2x1, df_dx2x2, df_dx2x3, df_dx2x4 }, { df_dx3x1, df_dx3x2, df_dx3x3, df_dx3x4 }, { df_dx4x1, df_dx4x2, df_dx4x3, df_dx4x4 } };

            return grad3;
        }

        // Optimization method for model 1
        public void Optim_model1(double[] x)
        {
            // initiate variables
            var xik = Matrix<double>.Build.Dense(kmax + 1, 2);

            xik[0, 0] = x[0]; // initiate
            xik[0, 1] = x[1]; // initiate

            // Initial Guess of x values
            double[] guessi = { xik[0, 0], xik[0, 1] };

            Console.WriteLine("Model (i): "
                + System.Environment.NewLine);

            for (int i = 0; ;)
            {
                if (i < kmax)
                {
                    // parse the current value of x with the current guess.
                    guessi = new double[] { xik[i, 0], xik[i, 1] };

                    // Re-evaluate the functions, gradient and Hessian matrix with new x values
                    Double[] fnew = Function1(guessi, t, y_t);
                    Double[] dfnew = Gradient_f1(guessi, t, y_t);
                    Double[,] ddfnew = Hessian_f1(guessi, t, y_t);

                    Console.WriteLine("x1: " + xik[i, 0] + " x2: " + xik[i, 1] +
                        System.Environment.NewLine + " f(x): " + "(" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", "
                            + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", "
                            + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");

                    // Convert ddfnew an array into matrix form
                    var M1 = Matrix<double>.Build;
                    Matrix<double> ddfm1 = M1.DenseOfArray(ddfnew);

                    // Convert dfnew an array into vector form
                    var V1 = Vector<double>.Build;
                    Vector<double> dfm1 = V1.Dense(dfnew);

                    // Calculate the eigenvalues of the Hessian matrix
                    Evd<double> eigen = ddfm1.Evd();
                    Vector<Complex> eigenvalues = eigen.EigenValues;

                    if (Math.Abs(dfnew.Max()) < e & eigenvalues[0].Real >= e & eigenvalues[1].Real >= e)
                    {
                        Console.WriteLine("A local minimizer of f within desired accuracy is given by xk: (" + xik[0, 0] + "," + xik[0, 1] +
                            ") and the function value"+
                            System.Environment.NewLine + " F(xk) is (" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", "
                            + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", "
                            + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");
                        break;
                    }

                    else if (Math.Abs(ddfm1.Determinant()) < e)
                    {
                        Console.WriteLine("The Hessian of f is very close to being singular.The iterates may not be stable.");
                        break;
                    }

                    else if (Math.Abs(dfnew.Max()) < e & eigenvalues[0].Real <= -e & eigenvalues[1].Real <= -e)
                    {
                        Console.WriteLine("A local maximiser of f within desired accuracy is given  by xk: (" + xik[0, 0] + ", " + xik[0, 1]
                            + ") and the function value is F(xk) is (" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", "
                            + fnew[3] + ", " + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9]
                            + ", " + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");
                        break;
                    }
                    else if (Math.Abs(dfnew.Max()) < e & ((eigenvalues[0].Real >= e | eigenvalues[1].Real <= -e | eigenvalues[1].Real >= e | eigenvalues[0].Real <= -e)))
                    {
                        Console.WriteLine("A root of the gradient of f within desired accuracy is given by xk: " + xik[0, 0] + "," + xik[0, 1] + " and the function value is"
                            + System.Environment.NewLine + "f(xk): ( " +
                            +fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", " + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", "
                            + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", " + fnew[10] + ", " + fnew[11] + ", " + fnew[12] +
                            ") However, xk is neither a minimizer nor a maximizer.");
                        break;
                    }
                    else if (Math.Abs(dfnew.Max()) < e)
                    {
                        Console.WriteLine("A zero of the gradient of f within desired accuracy is given by xk and the function value is f(xk). " +
                            "However, the Hessian test is inconclusive.");
                        break;
                    }
                    i++;

                    // newton-rapson update
                    // Make a new estimated guess for the two x values
                    xik[i, 0] = xik[i - 1, 0] - (ddfm1.Inverse() * dfm1)[0];
                    xik[i, 1] = xik[i - 1, 1] - (ddfm1.Inverse() * dfm1)[1];

                }

                else
                {
                    Console.WriteLine("Terminated  due  to  the  iteration limit without achieving desired accuracy level.");
                    break;
                }

            }
        }

        // Optimization method for model 2
        public void Optim_model2(double[] x)
        {
            // initiate variables
            var xik = Matrix<double>.Build.Dense(kmax + 1, 3);

            xik[0, 0] = x[0]; // initiate
            xik[0, 1] = x[1]; // initiate
            xik[0, 2] = x[2]; // initiate

            // Initiate the initial x value guesses
            double[] guessi = { xik[0, 0], xik[0, 1], xik[0, 2] };

            Console.WriteLine("Model (ii): " +
                System.Environment.NewLine);

            for (int i = 0; ;)
            {
                if (i < kmax)
                {
                    // parse the current value of x with the current guess.
                    guessi = new double[] { xik[i, 0], xik[i, 1], xik[i,2] };

                    // Re-evaluate the function, Gradient and Hessian matrix for the new x values
                    Double[] fnew = Function2(guessi, t, y_t);
                    Double[] dfnew = Gradient_f2(guessi, t, y_t);
                    Double[,] ddfnew = Hessian_f2(guessi, t, y_t);

                    Console.WriteLine("x1: " + xik[i, 0] + " x2: " + xik[i, 1] + " x3: " + xik[i,2] +
                        System.Environment.NewLine + " f(x): " + "(" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", "
                            + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", "
                            + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");

                    // Convert the Hessian matrix in array form into matrix format
                    var M2 = Matrix<double>.Build;
                    Matrix<double> ddfm2 = M2.DenseOfArray(ddfnew);

                    // Convert the Gradient vector in array form into vector format
                    var V2 = Vector<double>.Build;
                    Vector<double> dfm2 = V2.Dense(dfnew);

                    // Calculate the eigenvalues of the Hessian matrix
                    Evd<double> eigen = ddfm2.Evd();
                    Vector<Complex> eigenvalues = eigen.EigenValues;

                    if (Math.Abs(dfnew.Max()) < e & eigenvalues[0].Real >= e & eigenvalues[1].Real >= e)
                    {
                        Console.WriteLine("A local minimizer of f within desired accuracy is given by xk: (" + xik[0, 0] + "," + xik[0, 1] + ", " + xik[0, 2] +
                            ") and the function value " + System.Environment.NewLine
                            + "F(xk): (" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", "
                            + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", "
                            + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");
                        break;
                    }

                    else if (Math.Abs(ddfm2.Determinant()) < e)
                    {
                        Console.WriteLine("The Hessian of f is very close to being singular.The iterates may not be stable.");
                        break;
                    }

                    else if (Math.Abs(dfm2.Max()) < e & eigenvalues[0].Real <= -e & eigenvalues[1].Real <= -e)
                    {
                        Console.WriteLine("A local maximiser of f within desired accuracy is given  by xk: (" + xik[0, 0] + "," + xik[0, 1] + ", " + xik[0, 2]
                            + ") and the function value is F(xk) is (" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", "
                            + fnew[3] + ", " + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9]
                            + ", " + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");
                        break;
                    }
                    else if (Math.Abs(dfm2.Max()) < e & ((eigenvalues[0].Real >= e | eigenvalues[1].Real <= -e | eigenvalues[1].Real >= e | eigenvalues[0].Real <= -e)))
                    {
                        Console.WriteLine("A root of the gradient of f within desired accuracy is given by xk: " + xik[0, 0] + "," + xik[0, 1] + ", " + xik[0, 2] + " and the function value is F(xk): " +
                            +fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", " + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", "
                            + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", " + fnew[10] + ", " + fnew[11] + ", " + fnew[12] +
                            "However, xk is neither a minimizer nor a maximizer.");
                        break;
                    }
                    else if (Math.Abs(dfm2.Max()) < e)
                    {
                        Console.WriteLine("A zero of the gradient of f within desired accuracy is given by xk and the function value is f(xk). " +
                            "However, the Hessian test is inconclusive.");
                        break;
                    }
                    i++;


                    // newton-rapson update
                    // Make a new guess of x values based on the Newton's formula
                    xik[i, 0] = xik[i - 1, 0] - (ddfm2.Inverse() * dfm2)[0];
                    xik[i, 1] = xik[i - 1, 1] - (ddfm2.Inverse() * dfm2)[1];
                    xik[i, 2] = xik[i - 1, 2] - (ddfm2.Inverse() * dfm2)[2];
                }

                
                else
                {
                    Console.WriteLine("Terminated  due  to  the  iteration limit without achieving desired accuracy level.");
                    break;
                }
            }
        }

        // Optimization method for Model 3
        public void Optim_model3(double[] x)
        {
            // initiate variables
            var xik = Matrix<double>.Build.Dense(kmax + 1, 4);

            xik[0, 0] = x[0]; // initiate
            xik[0, 1] = x[1]; // initiate
            xik[0, 2] = x[2]; // initiate
            xik[0, 3] = x[3]; // initiate

            // Initialize the original guesses for the four x values
            double[] guessi = { xik[0, 0], xik[0, 1], xik[0, 2], xik[0, 3] };

            Console.WriteLine("Model (iii): " +
                System.Environment.NewLine);

            for (int i = 0; ;) // Loop to get kmax number of iterations of x_k
            {
                if (i < kmax)
                {

                    // parse the current value of x with the current guess.
                    guessi = new double[] { xik[i, 0], xik[i, 1], xik[i, 2], xik[i, 3] };

                    // Re-evaluate the function, gradient, and Hessian matrix for the new values of x
                    Double[] fnew = Function3(guessi, t, y_t);
                    Double[] dfnew = Gradient_f3(guessi, t, y_t);
                    Double[,] ddfnew = Hessian_f3(guessi, t, y_t);

                    Console.WriteLine("x1: " + xik[i, 0] + " x2: " + xik[i, 1] +
                        System.Environment.NewLine + " F(xk): " + "(" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", "
                            + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", "
                            + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");


                    // Convert the Hessian matrix in array form into matrix format
                    var M3 = Matrix<double>.Build;
                    Matrix<double> ddfm3 = M3.DenseOfArray(ddfnew);

                    // Convert the Gradient vector currently as an array into vector format
                    var V3 = Vector<double>.Build;
                    Vector<double> dfm3 = V3.DenseOfArray(dfnew);

                    // Calculate the Eigenvalues of the Hessian Matrix
                    Evd<double> eigen = ddfm3.Evd();
                    Vector<Complex> eigenvalues = eigen.EigenValues;

                    if (Math.Abs(dfm3.Max()) < e & eigenvalues[0].Real >= e & eigenvalues[1].Real >= e)
                    {
                        Console.WriteLine("A local minimizer of f within desired accuracy is given by xk: (" + xik[0, 0] + "," + xik[0, 1] + ", " + xik[0, 2] + ", " + xik[0, 3]
                            + ") and the function value" +
                            System.Environment.NewLine + "F(xk): (" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", "
                            + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", "
                            + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");
                        break;
                    }

                    else if (Math.Abs(ddfm3.Determinant()) < e)
                    {
                        Console.WriteLine("The Hessian of f is very close to being singular.The iterates may not be stable.");
                        break;
                    }

                    else if (Math.Abs(dfm3.Max()) < e & eigenvalues[0].Real <= -e & eigenvalues[1].Real <= -e)
                    {
                        Console.WriteLine("A local maximiser of f within desired accuracy is given  by xk: (" + xik[0, 0] + "," + xik[0, 1] + ", " + xik[0, 2] + ", " + xik[0, 3]
                            + ") and the function value is " +
                            System.Environment.NewLine + "F(xk): (" + fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", "
                            + fnew[3] + ", " + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", " + fnew[7] + ", " + fnew[8] + ", " + fnew[9]
                            + ", " + fnew[10] + ", " + fnew[11] + ", " + fnew[12] + ").");
                        break;
                    }
                    else if (Math.Abs(dfm3.Max()) < e & ((eigenvalues[0].Real >= e | eigenvalues[1].Real <= -e | eigenvalues[1].Real >= e | eigenvalues[0].Real <= -e)))
                    {
                        Console.WriteLine("A root of the gradient of f within desired accuracy is given by xk: " + xik[0, 0] + "," + xik[0, 1] + ", " + xik[0, 2] + ", " + xik[0, 3]
                            + " and the function value is f(xk). " +
                            +fnew[0] + ", " + fnew[1] + ", " + fnew[2] + ", " + fnew[3] + ", " + fnew[4] + ", " + fnew[5] + ", " + fnew[6] + ", "
                            + fnew[7] + ", " + fnew[8] + ", " + fnew[9] + ", " + fnew[10] + ", " + fnew[11] + ", " + fnew[12] +
                            "However, xk is neither a minimizer nor a maximizer.");
                        break;
                    }
                    else if (Math.Abs(dfm3.Max()) < e)
                    {
                        Console.WriteLine("A zero of the gradient of f within desired accuracy is given by xk and the function value is f(xk). " +
                            "However, the Hessian test is inconclusive.");
                        break;
                    }
                    i++;

                    // newton-rapson update
                    // Make a new estimation of x values based on Newton's formula
                    xik[i, 0] = xik[i - 1, 0] - (ddfm3.Inverse() * dfm3)[0];
                    xik[i, 1] = xik[i - 1, 1] - (ddfm3.Inverse() * dfm3)[1];
                    xik[i, 2] = xik[i - 1, 2] - (ddfm3.Inverse() * dfm3)[2];
                    xik[i, 3] = xik[i - 1, 3] - (ddfm3.Inverse() * dfm3)[3];

                }

                else
                {
                    Console.WriteLine("Terminated  due  to  the  iteration limit without achieving desired accuracy level.");
                    break;
                }
            }
        }
    }
    public class DistanceMeasure
    {
        /// <summary>
        /// Returns the a modulus b
        /// </summary>
        /// <param name="a"> integer a in a mod b. </param>
        /// <param name="b"> integer b in a mod b. </param>
        public static int MathMod(int a, int b)
        {
            return (Math.Abs(a * b) + a) % b;
        }

        /// <summary>
        /// Returns a distance metric using the inputted points and the Euclidean norm
        /// </summary>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        public static Matrix<double> Euclidean(Matrix<double> Points)
        {
            int n = Points.RowCount; // number of points
            Matrix<Double> D = Matrix<double>.Build.Dense(n, n); //intialise n by n distance matrix

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    D[i, j] = Distance.Euclidean(Points.Row(i), Points.Row(j));//calculate entries of D
                    D[j, i] = D[i, j]; // D is symmetric
                }
            }
            return D;
        }

        /// <summary>
        /// Returns a distance metric using the inputted points and the Manhattan norm
        /// </summary>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        public static Matrix<double> Manhattan(Matrix<double> Points)
        {
            // method that returns the distance matrix for a set of points using the manhattan metric
            int n = Points.RowCount; // number of points
            Matrix<Double> D = Matrix<double>.Build.Dense(n, n); //initialise n by n distance matrix

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    D[i, j] = Distance.Manhattan(Points.Row(i), Points.Row(j)); // calculate entries of distance matrix
                    D[j, i] = D[i, j]; // D is symmetric
                }
            }
            return D;
        }

        /// <summary>
        /// Returns a distance metric using the inputted points and the Manhalanobis norm
        /// </summary>
        /// <param name="points"> Matrix where each row represents a point. </param>
        public static Matrix<double> Mahalanobis(Matrix<double> points)
        {
            // method that returns the distance matrix for a set of points using the mahalanobis metric
            int n = points.RowCount; // number of points
            Matrix<Double> D = Matrix<double>.Build.Dense(n, n); // initialise distance matrrix

            Matrix<double> Sinv = DistanceMeasure.Covariance(points); // determine the covariance matrix

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    D[i, j] = Math.Sqrt((points.Row(i) - points.Row(j)) * Sinv * (points.Row(i) - points.Row(j))); // calculate entries
                    D[j, i] = D[i, j]; // D is symmetric
                }
            }
            return D;
        }

        public static Matrix<double> Covariance(Matrix<double> points)
        {
            // function for calculating the covariance matrix
            int n = points.RowCount; // number of poitns
            Vector<double> mean = points.ColumnSums() / n; // mean position
            int d = points.ColumnCount; // dimensions
            Matrix<double> S = Matrix<double>.Build.Dense(d, d); // initialise covariance matrix
            for (int i = 0; i < n; i++)
            {
                S += (points.Row(i) - mean).ToColumnMatrix() * (points.Row(i) - mean).ToRowMatrix(); // calculate covariance
            }
            S /= n - 1;
            return S.Inverse();
        }
    }

    public class Algorithm : DistanceMeasure
    {
        /// <summary>
        /// Returns a list of a specified number of clusters, found using a specified distance function and the greedy algorithm
        /// </summary>
        /// <param name="numclsters"> Number of clusters to compute. </param>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating clusters. </param>
        public List<Cluster> Greedy(int numclsters, Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            Matrix<double> D = Distance(Points); // get distance matrix

            int numpts = Points.RowCount;
            int d = Points.ColumnCount; // dimension


            List<Cluster> clusters = new List<Cluster>(); // initiate list of instances of the Cluster class

            // initiate all points as clusters
            for (int i = 0; i < numpts; i++)
            {
                List<Vector<double>> newList1 = new List<Vector<double>>
                    {
                        Points.Row(i)
                    };

                List<int> newList2 = new List<int>
                    {
                       i+1
                    };

                Cluster newCluster = new Cluster { Cluster_points = newList1, Ave_point = Points.Row(i), Indexes = newList2 };
                clusters.Add(newCluster);
            }

            // iterate until number of clusters achieved
            while (numpts > numclsters)
            {
                double min = D.Enumerate(Zeros.AllowSkip).Minimum(); // find min distance
                Tuple<int, int, double> index = D.Find(min.Equals, Zeros.AllowSkip); // get index of pts that give min

                Vector<double> ave_point = (Points.Row(index.Item1) + Points.Row(index.Item2)) / 2; // calculate average
                //Matrix<double> newpoints = Matrix<double>.Build.Dense(points.RowCount - 1, d); // initiate new

                // update clusters
                clusters[index.Item1].Cluster_points.AddRange(clusters[index.Item2].Cluster_points);
                clusters[index.Item1].Ave_point = ave_point;
                clusters[index.Item1].Indexes.AddRange(clusters[index.Item2].Indexes);
                clusters.Remove(clusters[index.Item2]);
                numpts -= 1;

                // update points
                Points.SetRow(index.Item1, ave_point);
                Points = Points.RemoveRow(index.Item2);

                // update distance matrix
                D = Distance(Points);
            }

            return clusters;
        }

        /// <summary>
        /// Returns a list of a specified number of clusters, found using a specified distance function and the fixed algorithm
        /// </summary>
        /// <param name="numclsters"> Number of clusters to compute. </param>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating clusters. </param>
        public List<Cluster> Fixed(int numclsters, Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            // fixed algorithm for finding specified number of clusters num

            int numpts = Points.RowCount;
            int d = Points.ColumnCount; // dimensions

            // initiate list for storing clusters
            List<Cluster> clusterList = new List<Cluster>();

            // randomly shuffle an array and use the first numclstrs elements
            Random rnd = new Random();
            int[] arr = Enumerable.Range(0, numpts).OrderBy(c => rnd.Next()).ToArray();

            // add cluster centers to clusters
            for (int i = 0; i < numclsters; i++)
            {

                List<Vector<double>> newList1 = new List<Vector<double>>
                    {
                        Points.Row(arr[i])
                    };

                List<int> newList2 = new List<int>
                    {
                        arr[i] + 1
                    };

                Cluster newCluster = new Cluster { Cluster_points = newList1, Ave_point = Points.Row(arr[i]), Indexes = newList2 };
                clusterList.Add(newCluster);
            }

            bool change = true; // check for convergence
            while (change)
            {
                // clear labels
                foreach (Cluster cluster in clusterList)
                {
                    cluster.Cluster_points.Clear();
                    cluster.Indexes.Clear();
                }

                // assign all points to clusters
                for (int i = 0; i < numpts; i++)
                {
                    // initiate matrix of points for calulating distance
                    Matrix<double> newPoints = Matrix<double>.Build.Dense(numclsters + 1, d);

                    // set entries
                    for (int j = 0; j < numclsters; j++)
                    {
                        newPoints.SetRow(j, clusterList[j].Ave_point);
                    }
                    newPoints.SetRow(numclsters, Points.Row(i));

                    Matrix<double> D = Distance(newPoints); // distance matrix

                    // find which cluster to assign the point to
                    double min = D.Row(numclsters).Enumerate(Zeros.AllowSkip).Minimum();
                    Tuple<int, double> index = D.Row(numclsters).Find(min.Equals, Zeros.AllowSkip);

                    // add point to that cluster
                    clusterList[index.Item1].Cluster_points.Add(Points.Row(i));
                    clusterList[index.Item1].Indexes.Add(i + 1);
                }

                // initiate count for checking convergence
                int count = 0;
                foreach (Cluster cluster in clusterList)
                {
                    // check old averages against new ones
                    Vector<double> old_ave_point = cluster.Ave_point;
                    Vector<double> new_ave_point = Vector<double>.Build.Dense(d);

                    // fine new average
                    foreach (Vector<double> element in cluster.Cluster_points)
                    {
                        new_ave_point += element;
                    }
                    new_ave_point /= cluster.Cluster_points.Count;
                    cluster.Ave_point = new_ave_point;

                    // check for change
                    if (old_ave_point.Equals(new_ave_point))
                    {
                        count += 1;
                    }
                }
                // if no change then stop
                if (count == numclsters)
                {
                    change = false;
                }
            }

            return clusterList;
        }

    }

    public class TSP : DistanceMeasure
    {
        /// <summary>
        /// Returns the cost of a path, found using a specified distance function
        /// </summary>
        /// <param name="path"> Path to compute the cost of. </param>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating cost. </param>
        double TourCost(int[] path, Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            int n = Points.RowCount;
            Matrix<double> D = Distance(Points);
            double cost = 0;

            for (int i = 0; i < path.Length; i++)
            {
                cost += D[path[i] - 1, path[(i + 1) % n] - 1];
            }

            return cost;
        }

        /// <summary>
        /// Returns a cost and a path, found using a specified distance function and the Nearest Neighbour TSP algorithm
        /// starting from a set point
        /// </summary>
        /// <param name="startind"> Index of starting point. </param>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating cost. </param>
        public (double, int[]) NearestNeighbour(int startind, Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            int n = Points.RowCount;
            int start = startind;

            Matrix<double> D = Distance(Points);
            Matrix<double> copyD = D;

            double totalcost = 0;
            List<int> pathList = new List<int>
            {
                startind
            };

            while (n > 1)
            {
                double min = copyD.Row(start - 1).Enumerate(Zeros.AllowSkip).Minimum(); // find min distance
                Tuple<int, double> index = copyD.Row(start - 1).Find(min.Equals, Zeros.AllowSkip); // find index of point with min distance from start
                copyD.ClearColumn(start - 1); // set entries to zero so we don't go back to the same point


                pathList.Add(index.Item1 + 1); // add index to path
                totalcost += min; // add cost
                start = index.Item1 + 1; // update start

                n -= 1; // one point has been removed

                // when only one point is left we need to go back to the startind, add the cost of this
                if (n == 1)
                {
                    totalcost += D[startind - 1, start - 1];
                }
            }

            return (totalcost, pathList.ToArray());
        }

        /// <summary>
        /// Returns a cost and a path, found using a specified distance function and the best Nearest Neighbour TSP algorithm
        /// </summary>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating cost. </param>
        public (double, int[]) BestNearestNeighbour(Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            int n = Points.RowCount;
            double totalcost;
            double besttotalcost = double.MaxValue;
            int[] path;
            List<int> bestpathList = new List<int>();

            for (int i = 0; i < n; i++)
            {
                (totalcost, path) = NearestNeighbour(i + 1, Points, Distance);
                if (totalcost < besttotalcost)
                {
                    besttotalcost = totalcost;
                    bestpathList = path.ToList();
                }
            }

            return (besttotalcost, bestpathList.ToArray());
        }


        /// <summary>
        /// Returns a cost and a path, found using a specified distance function and the two edge exchange TSP algorithm
        /// starting with a set path
        /// </summary>
        /// <param name="path"> starting path. </param>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating cost. </param>
        public (double, int[]) TwoEdgeEx(int[] path, Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            int n = Points.RowCount;
            Matrix<double> D = Distance(Points);

            // initiate best cost
            double bestcost = TourCost(path, Points, Distance);

            List<int> OldP = path.OfType<int>().ToList();
            List<int> BestP = path.OfType<int>().ToList();
            int[] newpath;

            // iterate over path and check the cost of each path in the neighbourhood of th given path
            for (int i = 0; i < n - 2; i++)
            {
                for (int j = 2; j < (i == 0 ? n - 1 - i : n - i); j++)
                {
                    // check cost of new path
                    newpath = OldP.ToArray();
                    Array.Reverse(newpath, i, j);
                    double newcost = TourCost(newpath, Points, Distance);

                    if (newcost < bestcost)
                    {
                        // check best cost and best path
                        bestcost = newcost;
                        BestP = newpath.OfType<int>().ToList();
                    }
                }
            }
            return (bestcost, BestP.ToArray());
        }

        /// <summary>
        /// Returns a cost and a path, found using a specified distance function and a large scale TSP algorithm with clusters
        /// </summary>
        /// <param name="numclsters"> Number of clusters to use. </param>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating cost. </param>
        public (double, int[]) LargeScaleNN(int numclsters, Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            int n = Points.RowCount;
            int d = Points.ColumnCount; // dimensions

            // initiate new list of clusters and an instance of the cluster algorithm class 
            List<Cluster> clusterList = new List<Cluster>();
            Algorithm MakeClusters = new Algorithm();

            clusterList = MakeClusters.Fixed(numclsters, Points, Distance);

            // initiate variables for calculations
            List<int[]> pathList = new List<int[]>();
            double cost;
            double total_cost = 0;
            int[] path;
            int[] clusterpath;

            Matrix<double> Centers = Matrix<double>.Build.Dense(numclsters, d);

            // calculate path between cluster centers
            for (int i = 0; i < numclsters; i++)
            {
                Centers.SetRow(i, clusterList[i].Ave_point);
            }

            (_, clusterpath) = NearestNeighbour(numclsters, Centers, Distance);

            // calculate paths within each cluster
            foreach (Cluster cluster in clusterList)
            {
                int numpts = cluster.Cluster_points.Count;
                Matrix<double> Clusterpoints = Matrix<double>.Build.Dense(numpts, d);
                for (int i = 0; i < numpts; i++)
                {
                    Clusterpoints.SetRow(i, cluster.Cluster_points[i]);
                }
                (cost, path) = NearestNeighbour(numpts, Clusterpoints, Distance);
                pathList.Add(path);
                total_cost += cost;
            }

            // inititate list for storing indexes where we will join clusters
            List<int> joinIndexes = new List<int>();

            // calculate where to join clusters
            int clusterind = 0;

            // join clusters together
            List<int> newPath = new List<int>();

            foreach (int i in clusterpath)
            {
                double mincost = double.MaxValue;
                //double min_cost_in = double.MaxValue;
                int in_out_index = 0;
                int numpts = clusterList[i - 1].Cluster_points.Count;

                int ptsind = 0;
                foreach (int j in pathList[i - 1])
                {
                    Matrix<double> PointsOut = Matrix<double>.Build.Dense(2, d);
                    Vector<double> MidPoint = (clusterList[i - 1].Cluster_points[j - 1] + clusterList[i - 1].Cluster_points[pathList[i - 1][(ptsind + 1) % numpts] - 1]) / 2;
                    PointsOut.SetRow(0, clusterList[clusterpath[(clusterind + 1) % numclsters] - 1].Ave_point);
                    PointsOut.SetRow(1, MidPoint);

                    Matrix<double> Dout = Distance(PointsOut);

                    Matrix<double> PointsIn = Matrix<double>.Build.Dense(2, d);

                    PointsIn.SetRow(0, clusterList[clusterpath[MathMod(clusterind - 1, numclsters)] - 1].Ave_point);
                    PointsIn.SetRow(1, MidPoint);

                    Matrix<double> Din = Distance(PointsIn);

                    cost = Dout.Enumerate(Zeros.AllowSkip).Minimum() + Din.Enumerate(Zeros.AllowSkip).Minimum();

                    if (cost < mincost)
                    {
                        in_out_index = j;
                        mincost = cost;
                    }
                    ptsind += 1;
                }
                joinIndexes.Add(in_out_index);

                // join clusters together

                for (int j = Array.IndexOf(pathList[i - 1], in_out_index) + 1; j < pathList[i - 1].Length + Array.IndexOf(pathList[i - 1], in_out_index) + 1; j++)
                {
                    newPath.Add(clusterList[i - 1].Indexes[pathList[i - 1][j % pathList[i - 1].Length] - 1]);
                }

                clusterind += 1;
            }

            // calculate cost of joining clusters
            clusterind = 0;
            foreach (int i in clusterpath)
            {
                int numpts = clusterList[i - 1].Cluster_points.Count;
                Matrix<double> JoinPoints = Matrix<double>.Build.Dense(2, d);
                JoinPoints.SetRow(0, clusterList[i - 1].Cluster_points[(joinIndexes[clusterind]) % numpts]);
                JoinPoints.SetRow(1, clusterList[clusterpath[(clusterind + 1) % numclsters] - 1].Cluster_points[joinIndexes[(clusterind + 1) % numclsters] - 1]);

                Matrix<double> D = Distance(JoinPoints);
                total_cost += D.Enumerate(Zeros.AllowSkip).Minimum();
                clusterind += 1;
            }

            return (total_cost, newPath.ToArray());

        }

        /// <summary>
        /// Returns a cost and a path, found using a specified distance function and a large scale TSP best nearest neighbour algorithm with clusters
        /// </summary>
        /// <param name="numclsters"> Number of clusters to use. </param>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating cost. </param>
        public (double, int[]) LargeScaleBNN(int numclsters, Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            int n = Points.RowCount;
            int d = Points.ColumnCount; // dimensions

            // initiate new list of clusters and an instance of the cluster algorithm class 
            List<Cluster> clusterList = new List<Cluster>();
            Algorithm MakeClusters = new Algorithm();

            clusterList = MakeClusters.Fixed(numclsters, Points, Distance);

            // initiate variables for calculations
            List<int[]> pathList = new List<int[]>();
            double cost = 0;
            double total_cost = 0;
            int[] path;
            int[] clusterpath;

            Matrix<double> Centers = Matrix<double>.Build.Dense(numclsters, d);

            // calculate path between cluster centers
            for (int i = 0; i < numclsters; i++)
            {
                Centers.SetRow(i, clusterList[i].Ave_point);
            }

            (_, clusterpath) = BestNearestNeighbour(Centers, Distance);

            // calculate paths within each cluster
            foreach (Cluster cluster in clusterList)
            {
                int numpts = cluster.Cluster_points.Count;
                Matrix<double> Clusterpoints = Matrix<double>.Build.Dense(numpts, d);
                for (int i = 0; i < numpts; i++)
                {
                    Clusterpoints.SetRow(i, cluster.Cluster_points[i]);
                }
                (cost, path) = BestNearestNeighbour(Clusterpoints, Distance);
                pathList.Add(path);
                total_cost += cost;
            }

            // inititate list for storing indexes where we will join clusters
            List<int> joinIndexes = new List<int>();

            // calculate where to join clusters
            int clusterind = 0;

            // join clusters together
            List<int> newPath = new List<int>();

            foreach (int i in clusterpath)
            {
                double mincost = double.MaxValue;
                //double min_cost_in = double.MaxValue;
                int in_out_index = 0;
                int numpts = clusterList[i - 1].Cluster_points.Count;

                int ptsind = 0;
                foreach (int j in pathList[i - 1])
                {
                    Matrix<double> PointsOut = Matrix<double>.Build.Dense(2, d);
                    Vector<double> MidPoint = (clusterList[i - 1].Cluster_points[j - 1] + clusterList[i - 1].Cluster_points[pathList[i - 1][(ptsind + 1) % numpts] - 1]) / 2;
                    PointsOut.SetRow(0, clusterList[clusterpath[(clusterind + 1) % numclsters] - 1].Ave_point);
                    PointsOut.SetRow(1, MidPoint);

                    Matrix<double> Dout = Distance(PointsOut);

                    Matrix<double> PointsIn = Matrix<double>.Build.Dense(2, d);

                    PointsIn.SetRow(0, clusterList[clusterpath[MathMod(clusterind - 1, numclsters)] - 1].Ave_point);
                    PointsIn.SetRow(1, MidPoint);

                    Matrix<double> Din = Distance(PointsIn);

                    cost = Dout.Enumerate(Zeros.AllowSkip).Minimum() + Din.Enumerate(Zeros.AllowSkip).Minimum();

                    if (cost < mincost)
                    {
                        in_out_index = j;
                        mincost = cost;
                    }
                    ptsind += 1;
                }
                joinIndexes.Add(in_out_index);

                // join clusters together

                for (int j = Array.IndexOf(pathList[i - 1], in_out_index) + 1; j < pathList[i - 1].Length + Array.IndexOf(pathList[i - 1], in_out_index) + 1; j++)
                {
                    newPath.Add(clusterList[i - 1].Indexes[pathList[i - 1][j % pathList[i - 1].Length] - 1]);
                }

                clusterind += 1;
            }

            // calculate cost of joining clusters
            clusterind = 0;
            foreach (int i in clusterpath)
            {
                int numpts = clusterList[i - 1].Cluster_points.Count;
                Matrix<double> JoinPoints = Matrix<double>.Build.Dense(2, d);
                JoinPoints.SetRow(0, clusterList[i - 1].Cluster_points[(joinIndexes[clusterind]) % numpts]);
                JoinPoints.SetRow(1, clusterList[clusterpath[(clusterind + 1) % numclsters] - 1].Cluster_points[joinIndexes[(clusterind + 1) % numclsters] - 1]);

                Matrix<double> D = Distance(JoinPoints);
                total_cost += D.Enumerate(Zeros.AllowSkip).Minimum();
                clusterind += 1;
            }

            return (total_cost, newPath.ToArray());

        }

        /// <summary>
        /// Returns a cost and a path, found using a specified distance function and a large scale TSP two edge exchange algorithm with clusters
        /// </summary>
        /// <param name="numclsters"> Number of clusters to use. </param>
        /// <param name="Points"> Matrix where each row represents a point. </param>
        /// <param name="Distance"> Metric to use in calculating cost. </param>

        public (double, int[]) LargeScaleEE(int numclsters, Matrix<double> Points, Func<Matrix<double>, Matrix<double>> Distance)
        {
            int n = Points.RowCount;
            int d = Points.ColumnCount; // dimensions

            // initiate new list of clusters and an instance of the cluster algorithm class 
            List<Cluster> clusterList = new List<Cluster>();
            Algorithm MakeClusters = new Algorithm();

            clusterList = MakeClusters.Fixed(numclsters, Points, Distance);

            // initiate variables for calculations
            List<int[]> pathList = new List<int[]>();
            double cost;
            double total_cost = 0;
            int[] path;
            int[] clusterpath;

            Matrix<double> Centers = Matrix<double>.Build.Dense(numclsters, d);

            // calculate path between cluster centers
            for (int i = 0; i < numclsters; i++)
            {
                Centers.SetRow(i, clusterList[i].Ave_point);
            }

            (_, clusterpath) = NearestNeighbour(numclsters, Centers, Distance);

            (_, clusterpath) = TwoEdgeEx(clusterpath, Centers, Distance);
            // calculate paths within each cluster
            foreach (Cluster cluster in clusterList)
            {
                int numpts = cluster.Cluster_points.Count;
                Matrix<double> Clusterpoints = Matrix<double>.Build.Dense(numpts, d);
                for (int i = 0; i < numpts; i++)
                {
                    Clusterpoints.SetRow(i, cluster.Cluster_points[i]);
                }
                (_, path) = NearestNeighbour(numpts, Clusterpoints, Distance);
                (cost, path) = TwoEdgeEx(path, Clusterpoints, Distance);
                pathList.Add(path);
                total_cost += cost;
            }

            // inititate list for storing indexes where we will join clusters
            List<int> joinIndexes = new List<int>();

            // calculate where to join clusters
            int clusterind = 0;

            // join clusters together
            List<int> newPath = new List<int>();

            foreach (int i in clusterpath)
            {
                double mincost = double.MaxValue;
                //double min_cost_in = double.MaxValue;
                int in_out_index = 0;
                int numpts = clusterList[i - 1].Cluster_points.Count;

                int ptsind = 0;
                foreach (int j in pathList[i - 1])
                {
                    Matrix<double> PointsOut = Matrix<double>.Build.Dense(2, d);
                    Vector<double> MidPoint = (clusterList[i - 1].Cluster_points[j - 1] + clusterList[i - 1].Cluster_points[pathList[i - 1][(ptsind + 1) % numpts] - 1]) / 2;
                    PointsOut.SetRow(0, clusterList[clusterpath[(clusterind + 1) % numclsters] - 1].Ave_point);
                    PointsOut.SetRow(1, MidPoint);

                    Matrix<double> Dout = Distance(PointsOut);

                    Matrix<double> PointsIn = Matrix<double>.Build.Dense(2, d);

                    PointsIn.SetRow(0, clusterList[clusterpath[MathMod(clusterind - 1, numclsters)] - 1].Ave_point);
                    PointsIn.SetRow(1, MidPoint);

                    Matrix<double> Din = Distance(PointsIn);

                    cost = Dout.Enumerate(Zeros.AllowSkip).Minimum() + Din.Enumerate(Zeros.AllowSkip).Minimum();

                    if (cost < mincost)
                    {
                        in_out_index = j;
                        mincost = cost;
                    }
                    ptsind += 1;
                }
                joinIndexes.Add(in_out_index);

                // join clusters together

                for (int j = Array.IndexOf(pathList[i - 1], in_out_index) + 1; j < pathList[i - 1].Length + Array.IndexOf(pathList[i - 1], in_out_index) + 1; j++)
                {
                    newPath.Add(clusterList[i - 1].Indexes[pathList[i - 1][j % pathList[i - 1].Length] - 1]);
                }

                clusterind += 1;
            }

            // calculate cost of joining clusters
            clusterind = 0;
            foreach (int i in clusterpath)
            {
                int numpts = clusterList[i - 1].Cluster_points.Count;
                Matrix<double> JoinPoints = Matrix<double>.Build.Dense(2, d);
                JoinPoints.SetRow(0, clusterList[i - 1].Cluster_points[(joinIndexes[clusterind]) % numpts]);
                JoinPoints.SetRow(1, clusterList[clusterpath[(clusterind + 1) % numclsters] - 1].Cluster_points[joinIndexes[(clusterind + 1) % numclsters] - 1]);

                Matrix<double> D = Distance(JoinPoints);
                total_cost += D.Enumerate(Zeros.AllowSkip).Minimum();
                clusterind += 1;
            }

            return (total_cost, newPath.ToArray());

        }
    }

    public class Cluster
    {
        // class for saving features of a cluster
        public List<Vector<double>> Cluster_points { get; set; }
        public Vector<double> Ave_point { get; set; }

        public List<int> Indexes { get; set; }
    }

    public class ReadWriteFile
    {
        // Methods for reading and writing generated files
        public static Matrix<double> Read(string stream)
        {
            StreamReader reader = new StreamReader(stream);
            List<string> listA = new List<String>();
            List<string> listB = new List<String>();
            List<string> listC = new List<String>();
            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine();
                if (!String.IsNullOrWhiteSpace(line))
                {
                    string[] values = line.Split(',');
                    if (values.Length >= 3)
                    {
                        listA.Add(values[0]);
                        listB.Add(values[1]);
                        listC.Add(values[2]);
                    }
                }
            }
            reader.Close();

            Matrix<double> points = Matrix<double>.Build.Dense(listA.Count - 1, 2);
            for (int i = 0; i < listA.Count - 1; i++)
            {
                points[i, 0] = Convert.ToDouble(listB[i + 1]);
                points[i, 1] = Convert.ToDouble(listC[i + 1]);
            }

            return points;

        }

        public static void WritePoints(Matrix<double> Points, string filepath)
        {
            using (StreamWriter w = new StreamWriter(new FileStream(filepath, FileMode.Create, FileAccess.Write)))
            {
                for (int i = 0; i < Points.RowCount; i++)
                {
                    Console.WriteLine("hit");
                    Vector<double> row = Points.Row(i);
                    string line = string.Format("{0}", row[0]);
                    for (int j = 1; j < Points.ColumnCount; j++)
                    {
                        line.Concat(string.Format(",{0}", row[j]));
                    }
                    w.WriteLine(line);
                    w.Flush();
                }
                w.Close();
            }
        }

        public static void WriteClusters(List<Cluster> clusterList, string filepath)
        {
            using (StreamWriter w = new StreamWriter(new FileStream(filepath, FileMode.Create, FileAccess.Write)))
            {
                for (int i = 0; i < clusterList.Count; i++)
                {
                    for (int j = 0; j < clusterList[i].Cluster_points.Count; j++)
                    {
                        Vector<double> row = clusterList[i].Cluster_points[j];
                        string line = string.Format("{0} ,{1},{2}", i + 1, clusterList[i].Indexes[j], row[0]);

                        for (int k = 1; k < row.Count; k++)
                        {
                            line += string.Format(",{0}", row[k]);
                        }
                        w.WriteLine(line);
                        w.Flush();
                    }
                }
                w.Close();
            }
        }

        public static void WritePath(int[] path, string filepath)
        {
            using (StreamWriter w = new StreamWriter(new FileStream(filepath, FileMode.Create, FileAccess.Write)))
            {
                for (int i = 0; i < path.Length; i++)
                {
                    string line = string.Format("{0}", path[i]);
                    w.WriteLine(line);
                    w.Flush();
                }
                w.Close();
            }
        }
    }

    class MainClass
    {
        public static void Main(string[] args)
        {
            // Initial guesses for the x parameters for model 1, 2 and 3 respectively named x1, x2, and x3
            double[] x1 = { -5, - 0.5};
            double[] x2 = { 1.3, 0.3, 2.4 };
            double[] x3 = { 1, 1, 1, 1 };

            NewtonsMethod test1 = new NewtonsMethod();

            test1.Optim_model1(x1);
            test1.Optim_model2(x2);
            test1.Optim_model3(x3);

            // uncomment and run with paths
            /// Task 2 ////
            /// Add own path names 
            //Matrix<double> Points1 = ReadWriteFile.Read(@"C:\Users\s1134560\Documents\OOPA\Assignment3\Project3\Project3\cluster2.csv");
            //Matrix<double> Points2 = ReadWriteFile.Read(@"C:\Users\s1134560\Documents\OOPA\Assignment3\Project3\Project3\cluster4.csv");

            //////// testing performance of greedy against fixed /////
            //Algorithm test = new Algorithm();

            //List<Cluster> clusterList = new List<Cluster>();

            //Stopwatch stopwatch = Stopwatch.StartNew(); //creates and start the instance of Stopwatch
            //clusterList = test.Greedy(5, Points1, test.Euclidean);
            //stopwatch.Stop();
            //Console.WriteLine(string.Format("Time taken to run the greedy algorithm on cluster2.csv to find 5 clusters is {0} ms.", stopwatch.ElapsedMilliseconds));

            ////ReadWriteFile.WriteClusters(clusterList, @"C:\Users\s1134560\Documents\OOPA\Assignment3\Project3\Project3\cluster2_greedy.csv");

            //stopwatch.Reset(); //resets Stopwatch
            //stopwatch.Start();
            //clusterList = test.Fixed(5, Points1, test.Euclidean);
            //stopwatch.Stop();
            //Console.WriteLine(string.Format("Time taken to run the fixed cluster algorithm on cluster2.csv to find 5 clusters is {0} ms.", stopwatch.ElapsedMilliseconds));

            ////ReadWriteFile.WriteClusters(clusterList, @"C:\Users\s1134560\Documents\OOPA\Assignment3\Project3\Project3\cluster2_fixed.csv");

            //Console.WriteLine();

            //stopwatch.Reset(); //resets Stopwatch
            //stopwatch.Start();
            //clusterList = test.Greedy(5, Points2, test.Euclidean);
            //stopwatch.Stop();
            //Console.WriteLine(string.Format("Time taken to run the greedy algorithm on cluster4.csv to find 5 clusters is {0} ms.", stopwatch.ElapsedMilliseconds));

            ////ReadWriteFile.WriteClusters(clusterList, @"C:\Users\s1134560\Documents\OOPA\Assignment3\Project3\Project3\cluster4_greedy.csv");

            //stopwatch.Reset(); //resets Stopwatch
            //stopwatch.Start();
            //clusterList = test.Fixed(5, Points2, test.Euclidean);
            //stopwatch.Stop();
            //Console.WriteLine(string.Format("Time taken to run the fixed cluster algorithm on cluster4.csv to find 5 clusters is {0} ms.", stopwatch.ElapsedMilliseconds));

            ////ReadWriteFile.WriteClusters(clusterList, @"C:\Users\s1134560\Documents\OOPA\Assignment3\Project3\Project3\cluster4_fixed.csv");

            /// Task 3///
            /// Add own path names
            /// 
            //TSP test2 = new TSP();
            //double cost;
            //int number = 15;

            //Matrix<double> Cities1 = ReadWriteFile.Read(@"C:\Users\s1134560\Documents\OOPA\Assignment3\Project3\Project3\tsp1.csv");
            //Matrix<double> Cities2 = ReadWriteFile.Read(@"C:\Users\s1134560\Documents\OOPA\Assignment3\Project3\Project3\tsp2.csv");

            //(cost, _) = test2.LargeScaleNN(10, Cities1, DistanceMeasure.Euclidean);

            //Console.WriteLine(string.Format("Cost of nearest neighbour in cities 1 with {0} clusters: {1}", number, cost));

            //TSP test3 = new TSP();

            //(cost, _) = test3.LargeScaleBNN(number, Cities1, DistanceMeasure.Euclidean);

            //Console.WriteLine(string.Format("Cost of best nearest neighbour in cities 1 with {0} clusters: {1}", number, cost));

            //TSP test4 = new TSP();

            //(cost, _) = test4.LargeScaleEE(number, Cities1, DistanceMeasure.Euclidean);

            //Console.WriteLine(string.Format("Cost of two edges exchange in cities 1 with {0} clusters: {1}", number, cost));

            //TSP test5 = new TSP();

            //(cost, _) = test5.NearestNeighbour(1, Cities1, DistanceMeasure.Euclidean);

            //Console.WriteLine(string.Format("Cost of nearest neighbour in cities without clusters: {0}", cost));

            //TSP test6 = new TSP();

            //(cost, _) = test6.LargeScaleNN(number, Cities2, DistanceMeasure.Euclidean);

            //Console.WriteLine(string.Format("Cost of nearest neighbour in cities 2 with {0} clusters: {1}", number, cost));

            //TSP test7 = new TSP();

            //(cost, _) = test7.NearestNeighbour(1, Cities2, DistanceMeasure.Euclidean);

            //Console.WriteLine(string.Format("Cost of nearest neighbour in cities 2 without clusters: {0}", cost));



            Console.ReadKey();

        }
    }
}