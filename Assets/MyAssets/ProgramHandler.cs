using UnityEngine;

using NeuralNetworkSystem;
using System.Collections.Generic;

public class ProgramHandler : MonoBehaviour {

    NeuralNetwork Network;
    NeuralNetworkTrainer Trainer;

    private void OnEnable() {
        CreateNetwork();
    }

    void Update() {
        if (Input.GetKeyDown(KeyCode.L)) {
            if (Network != null) RefreshNetwork();
        }
        if (Input.GetKeyDown(KeyCode.Tab)) Visualization.ToggleInfo();

        SelectCell();
    }

    void SelectCell() {
        if (!CameraHandler.MouseOnScreen()) return;
        if (!Input.GetMouseButtonDown(0)) return;

        Vector2 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);

        RaycastHit2D hit = Physics2D.Raycast(mousePos, Vector2.zero);

        if (hit.collider == null) return;

        ObjectVariables vars = hit.collider.gameObject.GetComponent<ObjectVariables>();
        if (vars == null) return;

        Visualization.Focus(vars.layer, vars.index);
    }

    void RefreshNetwork() {
        Visualization.Visualize(Network);
    }

    void CreateNetwork() {
        int[] layers = { 3, 2, 4 };
        //int[] layers = { 784, 16, 16, 10 };
        Network = new NeuralNetwork(layers);

        List<Data> data = new List<Data>();

        int q = 1000;
        for (int x = -1 * q; x <= 1 * q; x++) {
            for (int y = -1 * q; y <= 1 * q; y++) {
                for (int z = -1 * q; z <= 1 * q; z++) {
                    int label;
                    if (((x + y + z) > 0) && (x > 0)) label = 0;
                    if (((x + y + z) > 0) && (x <= 0)) label = 1;
                    if (((x + y + z) <= 0) && (y > 0)) label = 2;
                    else label = 3;

                    Vector value = new Vector(3);
                    value[0] = x;
                    value[1] = y;
                    value[2] = z;

                    data.Add(new Data(value, label));
                }
            }
        }

        Trainer = new NeuralNetworkTrainer(Network, data.ToArray(), 0.001f);
    }
}
