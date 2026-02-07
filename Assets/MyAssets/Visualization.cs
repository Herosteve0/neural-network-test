using UnityEngine;
using System;
using System.Collections.Generic;

public class Visualization : MonoBehaviour
{
    [SerializeField] GameObject neuron;
    [SerializeField] GameObject neuronconnection;
    [SerializeField] GameObject canvas;

    List<List<GameObject>> network;
    List<List<List<GameObject>>> connections;

    int network_layers;
    List<int> network_indexes;

    public float space = 10f;

    void Awake() {
        InitList(3, new List<int> {2, 4, 3});

        for (int layer = 0; layer < network_layers; layer++) {
            for (int index = 0; index < network_indexes[layer]; index++) {
                CreateNeuron(layer, index);
            }
        }

        for (int layer = 0; layer < network_layers-1; layer++) {
            for (int index = 0; index < network_indexes[layer]; index++) {
                for (int dest = 0; dest < network_indexes[layer+1]; dest++) {
                    CreateNeuronConnection(layer, index, dest);
                }
            }
        }
    }

    void InitList(int layers, List<int> indexes) {
        if (layers <= 0) { return; }
        if (indexes.Count < layers) { return; }

        network_layers = layers;
        network_indexes = indexes;

        network = new List<List<GameObject>>();
        List<GameObject> tmp;
        List<List<GameObject>> tmp1;

        for (int layer = 0; layer < layers; layer++) {
            tmp = new List<GameObject>();
            for (int index = 0; index < indexes[layer]; index++) {
                tmp.Add(null);
            }
            network.Add(tmp);
        }

        connections = new List<List<List<GameObject>>>();
        for (int layer = 0; layer < layers-1; layer++) {
            tmp1 = new List<List<GameObject>>();
            for (int index = 0; index < indexes[layer]; index++) {
                tmp = new List<GameObject>();
                for (int index_to = 0; index_to < indexes[layer+1]; index_to++) {
                    tmp.Add(null);
                }
                tmp1.Add(tmp);
            }
            connections.Add(tmp1);
        }
    }

    void CreateNeuron(int layer, int index) {
        if (network.Count < layer) { return; }
        if (network[layer].Count < index) { return; }

        GameObject go = Instantiate(neuron, canvas.transform);
        network[layer][index] = go;
        go.name = "Neuron [" + layer.ToString() + ", " + index.ToString() + "]";
        float x = layer;
        float y = (network_indexes[layer] + 1) / 2f - index - 1;
        go.transform.position = new Vector2(x, y) * space;
    }

    void CreateNeuronConnection(int layer, int index_from, int index_to) {
        if (network.Count < layer+1) { return; }
        if (network[layer].Count < index_from) { return; }
        if (network[layer+1].Count < index_to) { return; }

        GameObject go = Instantiate(neuronconnection, canvas.transform);
        connections[layer][index_from][index_to] = go;
        go.name = "Neuron Connection [" + layer.ToString() + ", " + index_from.ToString() + "] -> [" + (layer+1).ToString() + ", " + index_to.ToString() + "]";
        
        float rotation = (network_indexes[layer+1] - network_indexes[layer])/2f - index_to + index_from;
        float x = layer + 0.5f;
        float y = ((network_indexes[layer+1] + network_indexes[layer]) / 4f) - ((index_to + index_from) / 2f) - 0.5f;
        Vector2 center = new Vector2(x, y);
        float length = Mathf.Sqrt(Mathf.Pow(rotation, 2)+1);

        Transform body = go.transform.GetChild(0);

        go.transform.position = center * space;
        body.rotation = Quaternion.Euler(0f, 0f, Mathf.Atan(rotation) * 180f / Mathf.PI);
        body.localScale = new Vector3(length * space, 1f, 1f);
    }
}
