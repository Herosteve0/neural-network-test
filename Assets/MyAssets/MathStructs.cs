using System;
using UnityEngine;
using System.Text;

namespace NeuralNetworkSystem {
    public class Vector {
        public Vector(int length) {
            Length = length;
            Data = new float[length];
        }
        public Vector(float[] values) {
            Length = values.Length;
            Data = new float[Length];
            for (int i = 0; i < Length; i++) {
                Data[i] = values[i];
            }
        }

        public readonly int Length;
        public readonly float[] Data;

        public float this[int index] {
            get => Data[index];
            set => Data[index] = value;
        }

        public override string ToString() {
            StringBuilder r = new StringBuilder().AppendLine();

            r.Append("(");
            for (int i = 0; i < Length; i++) {
                r.Append(Data[i].ToString());
                if (i < Length - 1) r.Append(", ");
            }
            r.Append(")");
            return r.ToString();
        }


        public static Vector SingleValue(int length, int index, float value = 1f) {
            Vector R = new Vector(length);
            for (int i = 0; i < length; i++) R[i] = (i == index) ? value : 0;
            return R;
        }

        public static Vector Random(int length, float min = 0f, float max = 1f) {
            Vector R = new Vector(length);
            for (int i = 0; i < length; i++) R[i] = UnityEngine.Random.Range(min, max);
            return R;
        }


        private static void CompareLengths(Vector A, Vector B) {
            if (A.Length != B.Length) throw new Exception("Vectors have unequal lengths.");
        }
        private static void CompareLengths(Vector A, Vector B, Vector C) {
            CompareLengths(A, B);
            CompareLengths(A, C);
        }


        public Matrix Transpose() {
            Matrix R = new Matrix(1, Length);

            for (int i = 0; i < Length; i++) {
                R[0, i] = this[i];
            }

            return R;
        }

        public void Map(Func<float, float> f, Vector Out) {
            CompareLengths(this, Out);

            for (int i = 0; i < Data.Length; i++) {
                Out.Data[i] = f(Data[i]);
            }
        }
        public void Map(Func<float, float> f) {
            for (int i = 0; i < Data.Length; i++) {
                Data[i] = f(Data[i]);
            }
        }

        public void SoftMax(Vector Out) {
            CompareLengths(this, Out);
            float max = Max();

            float sum = 0f;
            for (int i = 0; i < Length; i++) {
                sum += Mathf.Exp(Data[i] - max);
            }

            for (int i = 0; i < Length; i++) {
                Out.Data[i] = Mathf.Exp(Data[i] - max) / sum;
            }
        }


        public static void Add(Vector A, Vector B, Vector Out) {
            CompareLengths(A, B, Out);
            for (int i = 0; i < A.Length; i++) {
                Out[i] = A[i] + B[i];
            }
        }
        public void Add(Vector A) {
            CompareLengths(this, A);
            for (int i = 0; i < Length; i++) {
                Data[i] += A.Data[i];
            }
        }

        public static void Sub(Vector A, Vector B, Vector Out) {
            CompareLengths(A, B, Out);
            for (int i = 0; i < A.Length; i++) {
                Out[i] = A[i] - B[i];
            }
        }

        public static void Scale(Vector A, float scaler, Vector Out) {
            for (int i = 0; i < A.Length; i++) {
                Out[i] = A[i] * scaler;
            }
        }
        public static void Scale(Vector A, float scaler) {
            for (int i = 0; i < A.Length; i++) {
                A[i] *= scaler;
            }
        }

        public static Vector Multiply(Matrix A, Vector B) {
            Vector R = new Vector(A.Rows);

            for (int i = 0; i < R.Length; i++) {
                float v = 0;
                for (int k = 0; k < A.Columns; k++) {
                    v += A[i, k] * B[k];
                }
                R[i] = v;
            }

            return R;
        }

        public Vector ElementMultiply(Vector A) {
            Vector R = new Vector(A.Length);
            for (int i = 0; i < R.Length; i++) {
                R[i] = this[i] * A[i];
            }

            return R;
        }

        public float Max() {
            float r = Data[0];
            for (int i = 1; i < Data.Length; i++) {
                r = Math.Max(r, Data[i]);
            }
            return r;
        }

        public int MaxIndex() {
            int r = 0;
            for (int i = 1; i < Data.Length; i++) {
                if (Data[r] < Data[i]) r = i;
            }
            return r;
        }
    }

    public class Matrix {
        public Matrix(int rows, int columns) {
            Rows = rows;
            Columns = columns;
            Data = new float[rows * columns];
        }

        public int Rows { get; }
        public int Columns { get; }

        public float[] Data;

        public virtual float this[int row, int column] {
            get => Data[row * Columns + column];
            set => Data[row * Columns + column] = value;
        }

        public static Matrix Random(int rows, int cols, float min = 0f, float max = 1f) {
            Matrix R = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    R[i, j] = UnityEngine.Random.Range(min, max);
                }
            }
            return R;
        }

        public override string ToString() {
            StringBuilder r = new StringBuilder().AppendLine();

            int[] max = new int[Columns];
            for (int j = 0; j < Columns; j++) {
                max[j] = this[0, j].ToString().Length;
                for (int i = 1; i < Rows; i++) {
                    max[j] = Math.Max(this[i, j].ToString().Length, max[j]);
                }
            }

            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Columns; j++) {
                    r.Append(this[i, j].ToString().PadRight(max[j]));
                    if (j < Columns - 1) r.Append("   |   ");
                }
                if (i < Rows - 1) r.AppendLine();
            }

            return r.ToString();
        }

        public virtual Matrix Transpose() {
            Matrix R = new Matrix(Columns, Rows);

            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Columns; j++) {
                    R[j, i] = this[i, j];
                }
            }

            return R;
        }

        public virtual void Map(Func<float, float> f, Matrix Out) {
            for (int i = 0; i < Data.Length; i++) {
                Out.Data[i] = f(Data[i]);
            }
        }
        public virtual void Map(Func<float, float> f) {
            for (int i = 0; i < Data.Length; i++) {
                Data[i] = f(Data[i]);
            }
        }

        public static Matrix operator +(Matrix A, Matrix B) {
            if (A.Rows != B.Rows) throw new Exception("Tried to add unequal Matrices (Rows).");
            if (A.Columns != B.Columns) throw new Exception("Tried to add unequal Matrices (Columns).");

            Matrix R = new Matrix(A.Rows, A.Columns);

            for (int i = 0; i < A.Rows; i++) {
                for (int j = 0; j < A.Columns; j++) {
                    R[i, j] = A[i, j] + B[i, j];
                }
            }

            return R;
        }

        public static Matrix operator -(Matrix A, Matrix B) {
            if (A.Rows != B.Rows) throw new Exception("Tried to add unequal Matrices (Rows).");
            if (A.Columns != B.Columns) throw new Exception("Tried to add unequal Matrices (Columns).");

            Matrix R = new Matrix(A.Rows, A.Columns);

            for (int i = 0; i < A.Rows; i++) {
                for (int j = 0; j < A.Columns; j++) {
                    R[i, j] = A[i, j] - B[i, j];
                }
            }

            return R;
        }

        public static Matrix operator *(Matrix A, Matrix B) {
            if (B.Rows != A.Columns) throw new Exception("Tried to multiple two Matrices with wrong sizes (Rows x Columns).");

            Matrix R = new Matrix(A.Rows, B.Columns);

            for (int i = 0; i < R.Rows; i++) {
                for (int j = 0; j < R.Columns; j++) {
                    float v = 0;
                    for (int k = 0; k < A.Columns; k++) {
                        v += A[i, k] * B[k, j];
                    }
                    R[i, j] = v;
                }
            }

            return R;
        }
        public static Matrix operator *(Vector A, Matrix B) {
            if (B.Rows != 1) throw new Exception("Tried to multiple Vector and Matrix with wrong sizes (Rows x Columns = 1).");

            Matrix R = new Matrix(A.Length, B.Columns);

            for (int i = 0; i < R.Rows; i++) {
                for (int j = 0; j < R.Columns; j++) {
                    R[i, j] = A[i] * B[0, j];
                }
            }

            return R;
        }
        public static Matrix operator *(Matrix A, float scaler) {
            Matrix R = new Matrix(A.Rows, A.Columns);

            for (int i = 0; i < R.Data.Length; i++) {
                R.Data[i] = A.Data[i] * scaler;
            }

            return R;
        }

        public static Matrix operator /(Matrix A, float scaler) {
            Matrix R = new Matrix(A.Rows, A.Columns);

            for (int i = 0; i < R.Data.Length; i++) {
                R.Data[i] = A.Data[i] / scaler;
            }

            return R;
        }

        public static void ElementMultiply(Matrix A, Matrix B, Matrix Out) {
            if (A.Rows != Rows) throw new Exception("Tried to element-wise multiply unequal Matrices (Rows).");
            if (A.Columns != Columns) throw new Exception("Tried to element-wise multiply unequal Matrices (Columns).");

            Matrix R = new Matrix(Rows, Columns);

            for (int i = 0; i < A.Rows; i++) {
                for (int j = 0; j < A.Columns; j++) {
                    R[i, j] = A[i, j] * this[i, j];
                }
            }

            return R;
        }

        public static void NeuronCalculation(Matrix W, Vector I, Vector B, Vector Out) {
            
        }
    }
}