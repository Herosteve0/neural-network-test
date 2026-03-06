using System;
using System.Text;

namespace NeuralNetworkSystem {
    public class Vector {
        public Vector(int length) {
            Length = length;
            Data = new float[length];
            for (int i = 0; i < length; i++) Data[i] = 0;
        }
        public Vector(float[] values) {
            Length = values.Length;
            Data = new float[Length];
            for (int i = 0; i < Length; i++) Data[i] = values[i];
        }

        public readonly int Length;
        public float[] Data;

        public float this[int index] {
            get => Data[index];
            set => Data[index] = value;
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

        public int MaxIndex() {
            int r = 0;
            for (int i = 1; i < Data.Length; i++) {
                if (Data[r] < Data[i]) r = i;
            }
            return r;
        }

        // Bad functions for Education only \/ \/ \/ \/ \/ \/ \/ (allocate too much memory, mathematically correct, but memory inefficient)

        public Vector Map(Func<float, float> func) {
            Vector R = new Vector(Length);

            for (int i = 0; i < Length; i++) {
                R[i] = func(this[i]);
            }
            return R;
        }

        public Matrix Transpose() {
            Matrix R = new Matrix(1, Length);

            for (int i = 0; i < Length; i++) {
                R[0, i] = this[i];
            }
            return R;
        }

        
        public static Vector operator +(Vector A, Vector B) {
            if (A.Length != B.Length) throw new Exception("Tried to two Vectors with unequal Lengths!");

            Vector R = new Vector(A.Length);

            for (int i = 0; i < R.Length; i++) {
                R[i] = A[i] + B[i];
            }
            return R;
        }
        public static Vector operator -(Vector A, Vector B) {
            if (A.Length != B.Length) throw new Exception("Tried to two Vectors with unequal Lengths!");

            Vector R = new Vector(A.Length);

            for (int i = 0; i < R.Length; i++) {
                R[i] = A[i] - B[i];
            }
            return R;
        }

        public static Vector operator *(Matrix A, Vector B) {
            if (A.Columns != B.Length) throw new Exception("Tried to multiply Matrix and Vector with unequal Columns x Length!");

            Vector R = new Vector(A.Rows);

            for (int i = 0; i < A.Rows; i++) {
                R[i] = 0f;
                for (int j = 0; j < A.Columns; j++) {
                    R[i] += A[i, j] * B[j]; 
                }
            }
            return R;
        }
        public static Vector operator *(Vector A, float scaler) {
            Vector R = new Vector(A.Length);

            for (int i = 0; i < A.Length; i++) {
                R[i] = A[i] * scaler;
            }
            return R;
        }

        public Vector ElementMultiplication(Vector A) {
            if (Length != A.Length) throw new Exception("Tried to do element multiplication on two Vectors with unequal Lengths!");

            Vector R = new Vector(Length);

            for (int i = 0; i < Length; i++) {
                R[i] = this[i] * A[i];
            }
            return R;
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

        public void Transpose(Matrix Out) {
            if (Rows != Out.Columns) throw new Exception("Tried to output transpose result to Matrix with unequal length (Rows - Columns)!");
            if (Columns != Out.Rows) throw new Exception("Tried to output transpose result to Matrix with unequal length (Columns - Rows)!");

            for (int i = 0; i < Rows; i++) {
                for (int j = 0; j < Columns; j++) {
                    Out[j, i] = this[i, j];
                }
            }
        }

        // Bad functions for Education only \/ \/ \/ \/ \/ \/ \/ (allocate too much memory, mathematically correct, but memory inefficient)

        public Matrix Transpose() {
            Matrix R = new Matrix(Columns, Rows);

            for (int i = 0; i < R.Rows; i++) {
                for (int j = 0; j < R.Columns; j++) {
                    R[i, j] = this[j, i];
                }
            }
            return R;
        }


        public static Matrix operator+(Matrix A, Matrix B) {
            if (A.Rows != B.Rows) throw new Exception("Tried to add two Matrices with unequal lengths (Rows - Columns)!");
            if (A.Columns != B.Columns) throw new Exception("Tried to add two Matrices with unequal lengths (Rows - Columns)!");

            Matrix R = new Matrix(A.Rows, A.Columns);

            for (int i = 0; i < R.Rows; i++) {
                for (int j = 0; j < R.Columns; j++) {
                    R[i, j] = A[i, j] + B[i, j];
                }
            }
            return R;
        }
        public static Matrix operator-(Matrix A, Matrix B) {
            if (A.Rows != B.Rows) throw new Exception("Tried to add two Matrices with unequal lengths (Rows - Columns)!");
            if (A.Columns != B.Columns) throw new Exception("Tried to add two Matrices with unequal lengths (Rows - Columns)!");

            Matrix R = new Matrix(A.Rows, A.Columns);

            for (int i = 0; i < R.Rows; i++) {
                for (int j = 0; j < R.Columns; j++) {
                    R[i, j] = A[i, j] - B[i, j];
                }
            }
            return R;
        }

        public static Matrix operator*(Vector A, Matrix B) {
            if (B.Rows != 1) throw new Exception("Tried to multiply Vector and Matrix with unequal lengths (Rows != 1)");

            Matrix R = new Matrix(A.Length, B.Columns);

            for (int i = 0; i < A.Length; i++) {
                for (int j = 0; j < B.Columns; j++) {
                    R[i, j] = A[i] * B[0, j];
                }
            }
            return R;
        }
        public static Matrix operator*(Matrix A, float scaler) {
            Matrix R = new Matrix(A.Rows, A.Columns);

            for (int i = 0; i < A.Rows; i++) {
                for (int j = 0; j < A.Columns; j++) {
                    R[i, j] = A[i, j] * scaler;
                }
            }
            return R;
        }

    }
}