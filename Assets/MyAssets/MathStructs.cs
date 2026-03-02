using System;
using System.Numerics;
using System.Text;
using UnityEngine;

namespace NeuralNetworkSystem {
    public class Vector {
        public Vector(int length) {
            Length = length;
            Data = new float[length];
            for (int  i = 0; i < length; i++) Data[i] = 0;
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

        public void Map(Func<float, float> f, Vector Out) {
            CompareLengths(this, Out);

            for (int i = 0; i < Data.Length; i++) {
                Out.Data[i] = f(Data[i]);
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

        public static void Sub(Vector A, Vector B, Vector Out) {
            CompareLengths(A, B, Out);
            int simd_width = Vector<float>.Count;

            int i = 0;
            for (; i <= Out.Length - simd_width; i += simd_width) {
                var v_a = new Vector<float>(A.Data, i);
                var v_b = new Vector<float>(B.Data, i);
                (v_a - v_b).CopyTo(Out.Data, i);
            }
            for (; i < Out.Length; i++) {
                Out[i] = A[i] - B[i];
            }
        }
        public void Sub(Vector A) {
            CompareLengths(this, A);
            int simd_width = Vector<float>.Count;

            int i = 0;
            for (; i <= Length - simd_width; i += simd_width) {
                var v_a = new Vector<float>(A.Data, i);
                var v = new Vector<float>(Data, i);
                (v - v_a).CopyTo(Data, i);
            }
            for (; i < Length; i++) {
                Data[i] -= A.Data[i];
            }
        }

        public void Scale(float scaler) {
            int simd_width = Vector<float>.Count;

            int i = 0;
            for (; i <= Length - simd_width; i += simd_width) {
                var v = new Vector<float>(Data, i);
                (v * scaler).CopyTo(Data, i);
            }
            for (; i < Length; i++) {
                Data[i] *= scaler;
            }
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

        public void Sub(Matrix A) {
            if (Rows != A.Rows) throw new Exception("Tried to sub Matrices with different Rows!");
            if (Columns != A.Columns) throw new Exception("Tried to sub Matrices with different Columns!");

            int simd_width = Vector<float>.Count;
            int length = Rows * Columns;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v_a = new Vector<float>(A.Data, i);
                var v = new Vector<float>(Data, i);
                (v - v_a).CopyTo(Data, i);
            }
            for (; i < length; i++) {
                Data[i] -= A.Data[i];
            }
        }
        public void Scale(float scaler) {
            int length = Rows * Columns;

            int simd_width = Vector<float>.Count;

            int i = 0;
            for (; i <= length - simd_width; i += simd_width) {
                var v = new Vector<float>(Data, i);
                (v * scaler).CopyTo(Data, i);
            }
            for (; i < length; i++) {
                Data[i] *= scaler;
            }
        }
    }
}