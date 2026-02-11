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
    float?[] NeuronXCache;
    float?[][] NeuronYCache;

    public float spacing = 17.5f;

    public bool ShowInfo;
    public Vector2Int FocusedCell;

    void OnEnable() {
        if (instance != null && instance != this) {
            Debug.Log("Tried to set second instance in Visualize.");
            return;
        }
        instance = this;
        ShowInfo = false;
        FocusedCell = new Vector2Int(-1, -1);
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

        Refresh();
    }

    void Initialize(NeuralNetwork network) {
        if (network.Layers.Length <= 0) { return; }
        Clear();
        Network = network;

        network_layers = network.Layers.Length;
        network_indices = new int[network_layers];
            
        int max = 0;
        foreach (int i in network_indices) {
            max = Math.Max(max, i);
        }

        spacing *= Mathf.Max(Mathf.Sqrt(max), 1f);

        CameraHandler.AdjustToNetwork(Network);

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

        NeuronXCache = new float?[network_layers];
        for (int i = 0; i < network_layers; i++) {
            NeuronXCache[i] = null;
        }
        NeuronYCache = new float?[network_layers][];
        for (int i = 0; i < network_layers; i++) {
            NeuronYCache[i] = new float?[network_indices[i]];
            for (int j = 0; j < network_indices[i]; j++) {
                NeuronYCache[i][j] = null;
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
        instance.spacing = Mathf.Max(instance.spacing + delta, 10f);
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

    public static void ToggleInfo() { ToggleInfo(!instance.ShowInfo); }
    public static void ToggleInfo(bool state) {
        instance.ShowInfo = state;
        instance.Refresh();
    }

    public static void Focus(int layer, int index) {
        if (instance.Network == null) return;
        if (layer < 0 || layer > instance.network_layers) return;
        if (index < 0 || index > instance.network_indices[layer]) return;

        Vector2Int v = new Vector2Int(layer, index);
        if (instance.FocusedCell == v) v = new Vector2Int(-1, -1);
        instance.FocusedCell = v;
        instance.Refresh();
    }

    void CreateNeuron(int layer, int index) {
        if (layer < 0 || layer > network_layers) return;
        if (index < 0 || index > network_indices[layer]) return;

        GameObject go = Instantiate(Neuron_Prefab, transform);
        Neurons[layer][index] = go;
        go.name = "Neuron [" + layer.ToString() + ", " + index.ToString() + "]";

        ObjectVariables objectVariables = go.AddComponent<ObjectVariables>();
        objectVariables.type = ObjectType.Neuron;
        objectVariables.layer = layer;
        objectVariables.index = index;

        PositionNeuron(layer, index);
    }

    float GetNeuronX(int layer) {
        if (NeuronXCache[layer] != null) return (float)NeuronXCache[layer];
        float x = layer;
        for (int i = 0; i < layer; i++) {
            x += Mathf.Log10(network_indices[i]);
        }
        NeuronXCache[layer] = x;
        return x;
    }
    float GetNeuronY(int layer, int index) {
        if (NeuronYCache[layer][index] != null) return (float)NeuronYCache[layer][index];
        float y = (network_indices[layer] - 1) / 2f - index;
        NeuronYCache[layer][index] = y;
        return y;
    }
    void PositionNeuron(int layer, int index) {
        float x = GetNeuronX(layer);
        float y = GetNeuronY(layer, index);
        Neurons[layer][index].transform.position = new Vector2(x, y) * spacing;
    }

    void CreateNeuronConnection(int layer, int index_from, int index_to) {
        if (layer < 0 || layer + 1 > network_layers) return;
        if (index_from < 0 || index_from > network_indices[layer]) return;
        if (index_to < 0 || index_to > network_indices[layer + 1]) return;

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
        float dx = Mathf.Log10(network_indices[layer]) + 1;
        float dy = (network_indices[layer + 1] - network_indices[layer]) / 2f - index_to + index_from;

        Vector2 center = (Neurons[layer][index_from].transform.position + Neurons[layer + 1][index_to].transform.position)/2f;
        float rotation = dy/dx;
        float length = Mathf.Sqrt(Mathf.Pow(dy, 2) + Mathf.Pow(dx, 2));

        Transform body = Connections[layer][index_from][index_to].transform.GetChild(0);

        Connections[layer][index_from][index_to].transform.position = center;
        body.rotation = Quaternion.Euler(0f, 0f, Mathf.Atan(rotation) * 180f / Mathf.PI);
        body.localScale = new Vector3(length * spacing, 1f, 1f);
    }

    void RefreshNeuron(int layer, int index) {
        float activation = Network.Layers[layer].Values[index, 0];

        Color c = new Color(activation, activation, activation, 1f);

        bool focused = (FocusedCell == new Vector2Int(layer, index)) || ((Math.Abs(FocusedCell.x - layer) == 1) && FocusedCell.x != -1);
        bool state = focused || ShowInfo;

        Neurons[layer][index].transform.GetChild(1).gameObject.SetActive(state);
        Neurons[layer][index].transform.GetChild(2).gameObject.SetActive(focused);
        if (state) {
            Neurons[layer][index].transform.GetChild(1).GetComponent<Text>().color = activation < 0.5f ? Color.white : Color.black;
            Neurons[layer][index].transform.GetChild(1).GetComponent<Text>().text = activation.ToString("F2");

            if (focused) {
                Neurons[layer][index].transform.GetChild(2).GetComponent<Text>().text = Network.Layers[layer].Bias[index, 0].ToString("F1");
                Neurons[layer][index].transform.GetChild(2).GetComponent<Text>().color = Color.white;
            }
        }

        if (!focused) c.a = (FocusedCell == new Vector2Int(-1, -1)) ? 1f : 0.5f;

        Neurons[layer][index].transform.GetChild(0).GetComponent<SpriteRenderer>().color = c;

    }

    void RefreshNeuronConnection(int layer, int index, int index_to) {
        float weight = Network.Layers[layer + 1].Weights[index_to, index];

        Color c = new Color(Math.Min(weight / 5f, 0f), 0f, Math.Max(weight / 5f, 0f), 1f);

        bool focused = (FocusedCell == new Vector2Int(layer + 1, index_to)) || (FocusedCell == new Vector2Int(layer, index));

        Connections[layer][index][index_to].transform.GetChild(1).gameObject.SetActive(focused);
        if (focused) {
            Connections[layer][index][index_to].transform.GetChild(1).GetComponent<Text>().text = weight.ToString("F2");
            Connections[layer][index][index_to].transform.GetChild(1).GetComponent<Text>().color = Color.white;
        } else {
            c.a = (FocusedCell == new Vector2Int(-1, -1)) ? 0.75f : 0.4f;
        }

        Connections[layer][index][index_to].transform.GetChild(0).GetComponent<SpriteRenderer>().color = c;
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
