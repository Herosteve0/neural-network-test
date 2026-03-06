using NeuralNetworkSystem;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;

public class DetailVisualization:MonoBehaviour {
    public Text network_details;
    public List<Text> graph_numbers;

    public NeuralNetwork Network;
    public NeuralNetworkTrainer Trainer;

    public List<float> Losses;
    GameObject[] points;

    [SerializeField] float multX = 0.925f;
    [SerializeField] float multY = 0.95f;
    [SerializeField] float LimitX = 1920;
    [SerializeField] float LimitY = 1080 - 79 * 5;

    [SerializeField] float point_size = 35f;
    [SerializeField] 

    public static DetailVisualization instance;

    private void OnEnable() {
        instance = this;
    }

    private void OnDisable() {
        if (points != null) {
            foreach (var go in points) {
                DestroyImmediate(go);
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

    static int compress_threshold = 2000;
    static float compress_power = 4f;
    public static void StoreLoss(float loss) {
        if (instance.Losses.Count >= compress_threshold) CompressLosses();
        instance.Losses.Add(loss);
    }

    public static void CompressLosses() {
        List<float> tmp = new List<float>();

        List<float>.Enumerator listhander = instance.Losses.GetEnumerator();
        for (int i = 0; i < instance.Losses.Count; i += 2) {
            float value = 0f;
            for (int j = 0; j < compress_power; j++) {
                value += listhander.Current;
                listhander.MoveNext();
            }
            tmp.Add(value / compress_power);
        }

        instance.Losses.Clear();
        instance.Losses = tmp;
    }

    public static void ClearLosses() {
        instance.Losses.Clear();
    }


    [SerializeField] int progressbars = 34;
    void PrintDetails() {
        string txt = $"Learning Rate: {Trainer.learning_rate}\n";
        txt += $"Training Batch Size: {Trainer.batchSize}, Cycles: {Trainer.cycles}\n";
        
        txt += $"\n";

        double time = Trainer.timeDelta;
        if (Trainer.isTraining) time *= (Trainer.TrainingAmount - Trainer.TrainingProgress)/Trainer.batchSize;
        else if (Trainer.isTesting) time *= Trainer.TestingAmount - Trainer.TestingProgress;
        int seconds = (int)time % 60;
        int minutes = ((int)time / 60) % 60;
        int hours = ((int)time / 60) / 60;
        txt += $"Estimated Time: ";
        if (hours > 0) txt += $"{hours:D2}:";
        if ((minutes > 0) || (hours > 0)) txt += $"{minutes:D2}:";
        txt += $"{seconds:D2}";
        txt += $"\n";

        if (Trainer.isTraining) {
            float d = Trainer.TrainingAmount / progressbars;
            txt += $"Training Progress:";
            for (int i = 0; i < progressbars; i++) {
                txt += Trainer.TrainingProgress > d * i ? "█" : "▒";
            }
        } else if (Trainer.isTesting) {
            float d = Trainer.TestingAmount / progressbars;
            txt += $"Testing Progress:";
            for (int i = 0; i < progressbars; i++) {
                txt += Trainer.TestingProgress > d * i ? "█" : "▒";
            }
        } else {
            txt += "Program is idle.";
        }

        network_details.text = txt;
    }

    Vector2 PointPosition(int index, float loss, float max, float min) {
        float offsetX = 1920 * (1 - multX);
        float offsetY = 1080 * (1 - multY);

        float x = offsetX + index * (LimitX * multX - offsetX * 0.25f) / (Losses.Count + 1);

        float y;
        if ((max != 0) && (max != min)) y = offsetY + (1080 * (multY - 1) + LimitY - offsetY) * (loss - min) / (max - min);
        else y = LimitY / 2f;

        return new Vector2(x, y);
    }
    void GraphLoss() {
        if (points != null) {
            foreach (var point in points) {
                DestroyImmediate(point);
            }
        }

        points = new GameObject[Losses.Count];
        float max = Mathf.Max(Losses.ToArray());
        float min = Mathf.Min(Losses.ToArray());
        Vector2 prePos = Vector2.zero;

        Texture2D tex = new Texture2D(1, 1) {
            filterMode = FilterMode.Point,
        };
        tex.SetPixel(0, 0, Color.white);
        tex.Apply();

        Vector2 size = Vector2.one * point_size;
        if (Losses.Count > 1) size /= Mathf.Log(Losses.Count);

        int index = 0;
        foreach (float loss in Losses) {
            Vector2 pos = PointPosition(index, loss, max, min);

            points[index] = new GameObject("Loss Graph Points");
            points[index].transform.SetParent(transform);

            points[index].AddComponent<RawImage>().texture = tex;

            points[index].transform.position = pos;
            points[index].GetComponent<RectTransform>().sizeDelta = size;

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
        for (int i = graph_numbers.Count; i > 0; i--) {
            if ((max - min) / (i - 1) > 0.15f) {
                use = i;
                break;
            }
        }


        for (int i = 0; i < graph_numbers.Count; i++) {
            graph_numbers[i].enabled = i < use;
            if (i >= use) continue;

            float value = min;
            if (use != 1) value += i * (max - min) / (use - 1);
            graph_numbers[i].text = value.ToString("F2");
            Vector3 pos = PointPosition(0, value, max, min);
            pos.x = (1920 * (1 - multX) + 96f) / 2f;
            graph_numbers[i].transform.position = pos;
        }
    }
}
