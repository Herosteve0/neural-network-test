using System;
using System.Text;

public struct Vector {
    public Vector(int length) {
        Length = length;
        Data = new float[length];
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

    public static Vector SingleValue(int length, int index) {
        Vector R = Vector.Zero(length);
        R[index] = 1;
        return R;
    }

    public override string ToString() {
        StringBuilder r = new StringBuilder().AppendLine();

        r.Append("(");
        for (int i = 0; i < Length; i++) {
            r.Append(Data[i].ToString());
            if (i < Length) r.Append(", ");
        }
        r.Append(")");
        return r.ToString();
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

    public static Vector operator *(Vector A, float B) {
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

    public Vector DotProduct(Vector A) {
        if (Length != A.Length) throw new Exception("Tried to calculate dot product of two Vectors with unequal lengths.");

        Vector R = new Vector(A.Length);
        for (int i = 0; i < R.Length; i++) {
            R[i] = this[i] * A[i];
        }

        return R;
    }
}

public struct Matrix {
    public Matrix(int rows, int columns) {
        Rows = rows;
        Columns = columns;
        Data = new float[rows * columns];
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

    public static Matrix VectorSingleValue(int length, int index) {
        Matrix r = new Matrix(length, 1);
        for (int i = 0; i < length; i++) {
            r[i, 0] = (i == index) ? 1 : 0;
        }
        return r;
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
    public static Matrix operator *(Matrix A, float scaler) {
        Matrix R = new Matrix(A.Rows, A.Columns);

        for (int i = 0; i < R.Rows; i++) {
            for (int j = 0; j < R.Columns; j++) {
                v += A[i, j] * scaler;
                R[i, j] = v;
            }
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