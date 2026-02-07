using System;
using System.Text;
using UnityEngine;

public struct Table {
    public Table(int rows, int Columns) {
        Rows = rows;
        Columns = Columns;
        Data = new double[rows * columns];
    }

    public int Rows { get; }
    public int Columns { get; }

    public double[] Data;

    public double this[int row, int column] {
        get => Data[row * Columns + column];
        set => Data[row * Columns + column] = value;
    }

    public static Table Identity(int n) {
        Table r = new Table(n, n);
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                r[i, j] = i == j ? 1 : 0;
            }
        }
        return r;
    }

    public static Table Zero(int n) {
        Table r = new Table(n, n);
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                r[i, j] = 0;
            }
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

    public Table Transpose() {
        Table R = new Table(Columns, Rows);
        
        int loop = Rows > Columns ? Rows : Columns;

        for (int i = 0; i < loop; i++) {
            for (int j = 0; j < loop; j++) {
                R[i, j] = this[j, i];
            }
        }

        return R;
    }

    public static Table operator +(Table A, Table B) {
        if (A.Rows != B.Rows) throw new Exception("Tried to add unequal Tables (Rows).");
        if (A.Columns != B.Columns) throw new Exception("Tried to add unequal Tables (Columns).");

        Table R = new Table(A.Rows, A.Columns);

        for (int i = 0; i < A.Rows; i++) { 
            for (int j = 0; j < A.Columns; j++) {
                R[i, j] = A[i, j] + B[i, j]; 
            }
        }

        return R;
    }

    public static Table operator *(Table A, Table B) {
        if (B.Rows != A.Columns) throw new Exception("Tried to multiple two tables with false sizes (Rows x Columns).");

        Table R = new Table(A.Rows, B.Columns);

        for (int i = 0; i < R.Rows; i++) {
            for (int j = 0; j < R.Columns; j++) {
                double v = 0;
                for (int k = 0; k < A.Columns; k++) {
                    v += A[i, k] * B[k, j];
                }
                R[i, j] = v;
            }
        }

        return R;
    }
}

public class NeuralNetwork : MonoBehaviour {
    void OnEnable() {
        Table A = new Table(2, 3);
        A[0, 0] = 1;
        A[0, 1] = 5;
        A[0, 2] = -7;
        A[1, 0] = 0;
        A[1, 1] = 8;
        A[1, 2] = 9;

        Debug.Log(A.Transpose());
    }
}