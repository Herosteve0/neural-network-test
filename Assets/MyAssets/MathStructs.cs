using System;
using System.Text;

public struct Vector {
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
    public Vector(Matrix matrix) {
        if (matrix.Rows > 1 && matrix.Columns > 1) throw new Exception("Tried converting 2D Matrix into a Vector! (Rows and Columns are both greater than 1.)");

        if (matrix.Rows == 1) {
            Length = matrix.Columns;
            Data = new float[Length];
            for (int i = 0; i < Length; i++) {
                Data[i] = matrix[0, i];
            }
        } else {
            Length = matrix.Rows;
            Data = new float[Length];
            for (int i = 0; i < Length; i++) {
                Data[i] = matrix[i, 0];
            }
        }
    }

    public int Length { get; }
    public float[] Data;

    public float this[int index] {
        get => Data[index];
        set => Data[index] = value;
    }

    public static Vector Zero(int length) {
        Vector R = new Vector(length);
        for (int i = 0; i < length; i++) { R[i] = 0; }
        return R;
    }

    public static Vector One(int length) {
        Vector R = new Vector(length);
        for (int i = 0; i < length; i++) { R[i] = 1; }
        return R;
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


    public Matrix Transpose() {
        Matrix R = new Matrix(1, Length);

        for (int i = 0; i < Length; i++) {
            R[0, i] = this[i];
        }

        return R;
    }

    public Vector Map(Func<float, float> f) {
        Vector R = new Vector(Length);

        for (int i = 0; i < Data.Length; i++) {
            R.Data[i] = f(Data[i]);
        }
        return R;
    }

    public Vector SoftMax() {
        Vector R = new Vector(Length);

        float max = Max();

        float sum = 0f;
        for (int i = 0; i < Length; i++) {
            sum += UnityEngine.Mathf.Exp(Data[i] - max);
        }

        for (int i = 0; i < Length; i++) {
            R[i] = UnityEngine.Mathf.Exp(Data[i] - max) / sum;
        }

        return R;
    }


    public static Vector operator +(Vector A, Vector B) {
        if (A.Length != B.Length) throw new Exception("Tried to add two Vectors with unequal lengths.");

        Vector R = new Vector(A.Length);
        for (int i = 0; i < R.Length; i++) {
            R[i] = A[i] + B[i];
        }

        return R;
    }

    public static Vector operator -(Vector A, Vector B) {
        if (A.Length != B.Length) throw new Exception("Tried to subtract two Vectors with unequal lengths.");

        Vector R = new Vector(A.Length);
        for (int i = 0; i < R.Length; i++) {
            R[i] = A[i] - B[i];
        }

        return R;
    }

    public static Vector operator *(Vector A, float scaler) {
        Vector R = new Vector(A.Length);

        for (int i = 0; i < R.Length; i++) {
            R[i] = A[i] * scaler;
        }

        return R;
    }
    public static Vector operator *(Matrix A, Vector B) {
        if (B.Length != A.Columns) throw new Exception("Tried to multiple Matrix and Vector with wrong sizes (Length x Columns).");

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

    public static Vector operator /(Vector A, float scaler) {
        Vector R = new Vector(A.Length);

        for (int i = 0; i < R.Length; i++) {
            R[i] = A[i] / scaler;
        }

        return R;
    }

    public Vector DotProduct(Vector A) {
        if (Length != A.Length) throw new Exception("Tried to calculate dot product of two Vectors with unequal lengths.");

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
    public float Min() {
        float r = Data[0];
        for (int i = 1; i < Data.Length; i++) {
            r = Math.Min(r, Data[i]);
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
    public float MinIndex() {
        int r = 0;
        for (int i = 1; i < Data.Length; i++) {
            if (Data[r] > Data[i]) r = i;
        }
        return r;
    }
}

public struct Matrix {
    public Matrix(int rows, int columns) {
        Rows = rows;
        Columns = columns;
        Data = new float[rows * columns];
        for (int i = 0; i < Data.Length; i++) {
            Data[i] = 0.0f;
        }
    }
    public Matrix(Vector v) {
        Rows = v.Length;
        Columns = 1;

        Data = new float[Rows];
        for (int i = 0; i < Rows; i++) {
            this[i, 0] = v[i];
        }
    }

    public int Rows { get; }
    public int Columns { get; }

    public float[] Data;

    public float this[int row, int column] {
        get => Data[row * Columns + column];
        set => Data[row * Columns + column] = value;
    }

    public static Matrix Identity(int n) {
        Matrix r = new Matrix(n, n);
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                r[i, j] = i == j ? 1 : 0;
            }
        }
        return r;
    }

    public static Matrix Zero(int n) {
        Matrix r = new Matrix(n, n);
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                r[i, j] = 0;
            }
        }
        return r;
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

        int[] min = new int[Columns];
        for (int j = 0; j < Columns; j++) {
            min[j] = this[0,j].ToString().Length;
            for (int i = 1; i < Rows; i++) {
                min[j] = Math.Min(this[i, j].ToString().Length, min[j]);
            }
        }

        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j < Columns; j++) {
                r.Append(this[i, j].ToString().PadRight(min[j]));
                if (j < Columns - 1) r.Append("   |   ");
            }
            if (i < Rows - 1) r.AppendLine();
        }

        return r.ToString();
    }

    public Matrix Transpose() {
        Matrix R = new Matrix(Columns, Rows);

        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j < Columns; j++) {
                R[j, i] = this[i, j];
            }
        }

        return R;
    }

    public Matrix Map(Func<float, float> f) {
        Matrix R = new Matrix(Rows, Columns);

        for (int i = 0; i < Data.Length; i++) {
            R.Data[i] = f(Data[i]);
        }
        return R;
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

    public Matrix ElementMultiply(Matrix A) {
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

    public Matrix ElementDivision(Matrix A) {
        if (A.Rows != Rows) throw new Exception("Tried to element-wise multiply unequal Matrices (Rows).");
        if (A.Columns != Columns) throw new Exception("Tried to element-wise multiply unequal Matrices (Columns).");

        Matrix R = new Matrix(Rows, Columns);

        for (int i = 0; i < A.Rows; i++) {
            for (int j = 0; j < A.Columns; j++) {
                R[i, j] = A[i, j] / this[i, j];
            }
        }

        return R;
    }
}