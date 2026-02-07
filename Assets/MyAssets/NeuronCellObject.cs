using System;
using UnityEngine;
using UnityEngine.UI;

public class NeuronCellObject : MonoBehaviour
{
    public Neuron identity;

    Transform transform;

    Vector2 value_text_offset = new Vector2(0f, 0f);
    Vector2 bias_text_offset = new Vector2(0f, -3.5f);

    GameObject value_go;
    GameObject bias_go;

    Text value_text;
    Text bias_text;

    void Awake() {
        transform = GetComponent<Transform>();

        value_go = transform.GetChild(1).gameObject;
        bias_go = transform.GetChild(2).gameObject;

        value_text = value_go.GetComponent<Text>();
        bias_text = bias_go.GetComponent<Text>();
    }

    void DisplayInfo() {
        value_text.text = identity.value.ToString();
        bias_text.text = identity.bias.ToString();
    }
}
