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

        if (!CameraHandler.MouseOnScreen()) return;
        if (!Input.GetMouseButtonDown(0)) return;

        PointerEventData data = new PointerEventData(EventSystem.current);
        data.position = Input.mousePosition;

        List<RaycastResult> results = new List<RaycastResult>();
        EventSystem.current.RaycastAll(data, results);

        Debug.Log("meow");
        foreach (RaycastResult r in results) {
            ObjectVariables vars = r.gameObject.GetComponent<ObjectVariables>();
            Debug.Log(vars);
            if (vars == null) continue;
            if (vars.type != ObjectType.Neuron) continue;
            Debug.Log($"Layer {vars.layer}, Index: {vars.index}");
        }
    }

    void CreateNetwork() {
        int[] layers = { 5, 3, 3, 4 };
        Network = new NeuralNetwork(layers);
    }

    void RefreshNetwork() {
        Visualization.Visualize(Network);
    }
}
