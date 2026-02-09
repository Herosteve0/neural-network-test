using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.UIElements;

public class Visualization: MonoBehaviour {

    [SerializeField] GameObject Neuron_Prefab;
    [SerializeField] GameObject NeuronConnection_Prefab;
    public static Visualization instance;

    NeuralNetwork Network;

    GameObject[][] Neurons;
    GameObject[][][] Connections;

    int network_layers;
    int[] network_indices;

    public float space = 17.5f;

    void OnEnable() {
        if (instance != null && instance != this) {
            Debug.Log("Tried to set second instance in Visualize.");
            return;
        }
        instance = this;
    }

    private void OnDisable() {
        Clear();
    }

    public static void Visualize(NeuralNetwork network) {
        if (instance.Network == null) instance.Visual(network);
        else instance.Refresh();
    }

    void Visual(NeuralNetwork network) {
        Initialize(network);

        for (int layer = 0; layer < network_layers; layer++) {
            for (int index = 0; index < network_indices[layer]; index++) {
                CreateNeuron(layer, index);
            }
        }

        for (int layer = 0; layer < network_layers - 1; layer++) {
            for (int index = 0; index < network_indices[layer]; index++) {
                for (int dest = 0; dest < network_indices[layer + 1]; dest++) {
                    CreateNeuronConnection(layer, index, dest);
                }
            }
        }
    }

    void Initialize(NeuralNetwork network) {
        if (network.Layers.Length <= 0) { return; }
        Clear();
        Network = network;

        network_layers = network.Layers.Length;
        network_indices = new int[network_layers];

        Neurons = new GameObject[network_layers][];
        for (int layer = 0; layer < network_layers; layer++) {
            network_indices[layer] = network.Layers[layer].NeuronNum;
            Neurons[layer] = new GameObject[network.Layers[layer].NeuronNum];
        }

        Connections = new GameObject[network_layers - 1][][];
        for (int layer = 0; layer < network_layers - 1; layer++) {
            Connections[layer] = new GameObject[network_indices[layer]][];
            for (int index = 0; index < network.Layers[layer].NeuronNum; index++) {
                Connections[layer][index] = new GameObject[network_indices[layer + 1]];
            }
        }
    }

    void Clear() {
        Network = null;

        if (Neurons != null) {
            foreach (var layer in Neurons) {
                foreach (var n in layer) {
                    if (n) Destroy(n);
                }
            }
        }

        if (Connections != null) {
            foreach (var layer in Connections) {
                foreach (var index in layer) {
                    foreach (var n in index) {
                        if (n) Destroy(n);
                    }
                }
            }
        }
    }

    public static void ChangeSpace(float delta) {
        instance.space = Mathf.Max(instance.space + delta, 10f);
        UpdatePosition();
    }

    public static void UpdatePosition() {
        if (instance.Network == null) return;

        for (int layer = 0; layer < instance.network_layers; layer++) {
            for (int index = 0; index < instance.network_indices[layer]; index++) {
                instance.PositionNeuron(layer, index);
                if (layer == instance.network_layers - 1) continue;
                for (int index_to = 0; index_to < instance.network_indices[layer+1]; index_to++) {
                    instance.PositionNeuronConnection(layer, index, index_to);
                }
            }
        }
    }

    void CreateNeuron(int layer, int index) {
        if (Neurons.Length < layer) { return; }
        if (Neurons[layer].Length < index) { return; }

        GameObject go = Instantiate(Neuron_Prefab, transform);
        Neurons[layer][index] = go;
        go.name = "Neuron [" + layer.ToString() + ", " + index.ToString() + "]";

        ObjectVariables objectVariables = go.AddComponent<ObjectVariables>();
        objectVariables.type = ObjectType.Neuron;
        objectVariables.layer = layer;
        objectVariables.index = index;

        PositionNeuron(layer, index);
    }

    void PositionNeuron(int layer, int index) {
        float x = layer;
        float y = (network_indices[layer] + 1) / 2f - index - 1;
        Neurons[layer][index].transform.position = new Vector2(x, y) * space;
    }

    void RefreshNeuron(int layer, int index) {
        float activation = Network.Layers[layer].Values[index, 0];

        Neurons[layer][index].transform.GetChild(0).GetComponent<SpriteRenderer>().color = new Color(activation, activation, activation, 1f);
        Neurons[layer][index].transform.GetChild(1).GetComponent<Text>().color = activation < 0.5f ? Color.white : Color.black;
        Neurons[layer][index].transform.GetChild(1).GetComponent<Text>().text = activation.ToString("F2");
        Neurons[layer][index].transform.GetChild(2).GetComponent<Text>().text = Network.Layers[layer].Bias[index, 0].ToString("F1");
    }

    void RefreshNeuronConnection(int layer, int index, int index_to) {
        float weight = Network.Layers[layer + 1].Weights[index_to, index];

        Connections[layer][index][index_to].transform.GetChild(0).GetComponent<SpriteRenderer>().color = new Color(Math.Min(weight / 5f, 0f), 0f, Math.Max(weight / 5f, 0f), 1f);
        Connections[layer][index][index_to].transform.GetChild(1).GetComponent<Text>().text = weight.ToString("F2");
        Connections[layer][index][index_to].transform.GetChild(1).GetComponent<Text>().color = Color.white;
    }

    void CreateNeuronConnection(int layer, int index_from, int index_to) {
        if (Neurons.Length < layer + 1) { return; }
        if (Neurons[layer].Length < index_from) { return; }
        if (Neurons[layer + 1].Length < index_to) { return; }

        GameObject go = Instantiate(NeuronConnection_Prefab, transform);
        Connections[layer][index_from][index_to] = go;
        go.name = "Neuron Connection [" + layer.ToString() + ", " + index_from.ToString() + "] -> [" + (layer + 1).ToString() + ", " + index_to.ToString() + "]";
        
        ObjectVariables objectVariables = go.AddComponent<ObjectVariables>();
        objectVariables.type = ObjectType.Connection;
        objectVariables.layer = layer;
        objectVariables.index = index_from;
        objectVariables.index_to = index_to;

        PositionNeuronConnection(layer, index_from, index_to);
    }

    void PositionNeuronConnection(int layer, int index_from, int index_to) {
        float rotation = (network_indices[layer + 1] - network_indices[layer]) / 2f - index_to + index_from;
        float x = layer + 0.5f;
        float y = ((network_indices[layer + 1] + network_indices[layer]) / 4f) - ((index_to + index_from) / 2f) - 0.5f;
        Vector2 center = new Vector2(x, y);
        float length = Mathf.Sqrt(Mathf.Pow(rotation, 2) + 1);

        Transform body = Connections[layer][index_from][index_to].transform.GetChild(0);

        Connections[layer][index_from][index_to].transform.position = center * space;
        body.rotation = Quaternion.Euler(0f, 0f, Mathf.Atan(rotation) * 180f / Mathf.PI);
        body.localScale = new Vector3(length * space, 1f, 1f);
    }

    public void Refresh() {
        if (Network == null) return;

        for (int layer = 0; layer < network_layers; layer++) {
            for (int index = 0; index < network_indices[layer]; index++) {
                RefreshNeuron(layer, index);
                if (layer == network_layers - 1) continue;
                for (int index_to = 0; index_to < network_indices[layer + 1]; index_to++) {
                    RefreshNeuronConnection(layer, index, index_to);
                }
            }
        }
    }
}
