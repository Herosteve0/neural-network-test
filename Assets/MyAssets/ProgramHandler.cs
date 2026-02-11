using UnityEngine;
using UnityEngine.EventSystems;
using System.Collections.Generic;

public class ProgramHandler : MonoBehaviour {

    NeuralNetwork Network;

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

    void CreateNetwork() {
        //int[] layers = { 5, 3, 3, 4 };
        int[] layers = { 784, 16, 16, 10 };
        Network = new NeuralNetwork(layers);
    }

    void RefreshNetwork() {
        Visualization.Visualize(Network);
    }
}
