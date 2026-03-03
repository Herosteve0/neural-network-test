using NeuralNetworkSystem;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using static UnityEditor.Experimental.GraphView.GraphView;

public class DetailVisualization:MonoBehaviour {
    public Text network_details;
    public List<Text> graph_numbers;

    public NeuralNetwork Network;
    public NeuralNetworkTrainer Trainer;

    public List<float> Losses;
    GameObject[] points;

    [SerializeField] float multX = 0.95f;
    [SerializeField] float multY = 0.95f;
    [SerializeField] float LimitX = 1920;
    [SerializeField] float LimitY = 1080 - 316;

    public static DetailVisualization instance;

    private void OnEnable() {
        instance = this;
    }

    private void OnDisable() {
        if (points != null) {
            foreach (var go in points) {
                Destroy(go);
            }
        }
    }

    public static void Initialize(NeuralNetwork network, NeuralNetworkTrainer trainer) {
        instance.Network = network;
        instance.Trainer = trainer;
        instance.Losses = new List<float>();
        Refresh();
    }

    public static void Refresh() {
        instance.GraphLoss();
        instance.UpdateLabels();
        instance.PrintDetails();
    }

    public static void StoreLoss(float loss) {
        instance.Losses.Add(loss);
    }

    public static void ClearLosses() {
        instance.Losses.Clear();
    }


    [SerializeField] int progressbars = 34;
    void PrintDetails() {
        string txt = $"Learning Rate: {Trainer.learning_rate}\n";
        txt += $"Training Batch Size: {Trainer.batchSize}, Cycles: {Trainer.cycles}\n";
        txt += $"\n";
        if (Trainer.isTraining) {
            float d = Trainer.TrainingAmount / progressbars;
            txt += $"Training Progress:";
            for (int i = 0; i < progressbars; i++) {
                txt += Trainer.TrainingProgress > d * i ? "|" : "·";
            }
            txt += "\n";
        } else {
            txt += $"Training completed.\n";
        }

        network_details.text = txt;
    }

    Vector2 PointPosition(int index, float loss, float max) {
        float offsetX = 1920 * (1 - multX);
        float offsetY = 1080 * (1 - multY);

        float x = offsetX + index * (LimitX * multX - offsetX * 0.25f) / (Losses.Count + 1);

        float y = offsetY;
        if (max != 0) y += (LimitY * multY - offsetY * 2) * loss / max;

        return new Vector2(x, y);
    }
    void GraphLoss() {
        if (points != null) {
            foreach (var point in points) {
                Destroy(point);
            }
        }

        points = new GameObject[Losses.Count];
        float max = Mathf.Max(Losses.ToArray());
        Vector2 prePos = Vector2.zero;

        Texture2D tex = new Texture2D(1, 1) {
            filterMode = FilterMode.Point,
        };
        tex.SetPixel(0, 0, Color.white);
        tex.Apply();

        int index = 0;
        foreach (float loss in Losses) {
            Vector2 pos = PointPosition(index, loss, max);

            points[index] = new GameObject("Loss Graph Points");
            points[index].transform.SetParent(transform);

            points[index].AddComponent<RawImage>().texture = tex;

            points[index].transform.position = pos;
            points[index].GetComponent<RectTransform>().sizeDelta = Vector2.one * (40f / Mathf.Log(Losses.Count));

            index++;
        }
    }
    void UpdateLabels() {
        if (Losses.Count == 0) {
            foreach (var go in graph_numbers) go.enabled = false;
            return;
        }
        float min = Losses[0];
        float max = Losses[0];
        foreach (float loss in Losses) {
            if (min > loss) min = loss;
            if (max < loss) max = loss;
        }

        int use = 1;
        for (int c = graph_numbers.Count; c > 0; c--) {
            if ((max - min) / (c - 1) > 0.15f) {
                use = c;
                break;
            }
        }


        int i = 0;
        for (; i < use; i++) {
            float value = min + i * (max - min) / (use - 1);
            graph_numbers[i].text = value.ToString("F2");
            Vector3 pos = PointPosition(0, value, max);
            pos.x = 1920 * (1 - multX);
            graph_numbers[i].transform.position = pos;
            graph_numbers[i].enabled = true;
        }
        for (; i < graph_numbers.Count; i++) {
            graph_numbers[i].enabled = false;
        }
    }
}
