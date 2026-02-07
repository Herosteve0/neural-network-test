using System.Collections.Generic;
using UnityEngine;

public class Neuron {

    public float bias;
    public float value;
    public float rawvalue;
    public List<NeuronConnection> connections;

    private float sigmoid(float value) { return 1f / (1f + Mathf.Exp(-value)); }

    public Neuron(float b = 0f) {
        bias = b;
        value = 0;
        rawvalue = 0;
        connections = new List<NeuronConnection>();
    }

    public void AdjustBiasBy(float value) { bias += value; }

    public float Output() {
        float r = bias;
        foreach (NeuronConnection i in connections) {
            r += i.Output();
        }
        return sigmoid(r);
    }

    public void UpdateData() {
        float r = bias;
        foreach (NeuronConnection n in connections) {
            r += n.Output();
        }
        rawvalue = r;
        value = sigmoid(rawvalue);
    }

    public float CalcCellValue(float input) {
        if (connections.Count == 0) return input;
        float r = bias;

        foreach (NeuronConnection neuron in connections) {
            r += neuron.CalcValue(input);
        }
        return sigmoid(r);
    }

    public void MakeConnection(List<Neuron> neurons) {
        foreach (Neuron n in neurons) {
            if (n == this) continue;
            connections.Add(new NeuronConnection(n, this));
        }
    }
}

public class NeuronConnection {

    public Neuron from;
    public Neuron to;
    public float weight;

    public NeuronConnection(Neuron nfrom, Neuron nto, float w = 1f) {
        from = nfrom;
        to = nto;
        weight = w;
    }

    public void AdjustWeightBy(float value) { weight += value; }

    public float Output() { return weight * from.value; }

    public float CalcValue(float input) { return weight * from.CalcCellValue(input); }
}