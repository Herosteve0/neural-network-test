using UnityEngine;

using NeuralNetworkSystem;
using System.Collections.Generic;
using System.Threading.Tasks;

public class ProgramHandler : MonoBehaviour {
    
    NeuralNetwork Network;
    NeuralNetworkTrainer Trainer;

    [SerializeField] bool state = false;
    int training_index;

    private void OnEnable() {
        CreateNetwork(state);
    }

    void Update() {
        if (Input.GetKeyDown(KeyCode.L)) {
            if (Network != null) RefreshNetwork();
        }
        if (Input.GetKeyDown(KeyCode.Tab)) Visualization.ToggleInfo();
        if (Input.GetKeyDown(KeyCode.M)) Trainer.MNIST_Training(100);
        //if (Input.GetKeyDown(KeyCode.Space)) {
        //    if (Input.GetKey(KeyCode.LeftShift)) Test(true);
        //}
        if (Input.GetKeyDown(KeyCode.N)) LargeTest();
        if (Input.GetKeyDown(KeyCode.B)) {
            if (Input.GetKey(KeyCode.LeftShift)) Visualization.instance.WeightDiffToImage();
            else Visualization.instance.WeightToImage();
        }



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

    async Task LargeTest() {
        MNISTDatabase database = new MNISTDatabase("Assets/StreamingAssets/MNIST/t10k-images.idx3-ubyte", "Assets/StreamingAssets/MNIST/t10k-labels.idx1-ubyte");

        Debug.Log($"Started testing on {database.Size} test samples.");
        int a = 0;
        for (int i = 0; i < database.Size; i += 1) {
            Data TestingData = database.ReadBatch(1)[0];
            Vector result = Network.Calculate(TestingData.data);
            if (result.MaxIndex() == TestingData.label) a++;
            if (i % 2500 == 0) await Task.Delay(1);
        }
        Debug.Log($"Testing complete with {(double)a / database.Size * 100}% accuracy. [{a}/{database.Size}]");
    }



    void CreateNetwork(bool meow) {
        if (meow) {
            int[] layers = { 3, 2, 4 };
            Network = new NeuralNetwork(layers);
        } else {
            int[] layers = { 784, 16, 16, 10 };
            Network = new NeuralNetwork(layers);

            Trainer = new NeuralNetworkTrainer(Network, 0.1f);
        }
    }
}
