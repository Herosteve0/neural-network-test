using System;
using Unity.VisualScripting;
using UnityEngine;

using NeuralNetworkSystem;

public class CameraHandler : MonoBehaviour
{
    Transform transform;
    Camera camera;
    static CameraHandler instance;

    [SerializeField] float move_speed = 25.0f;
    [SerializeField] float mouse_speed = 5f;
    [SerializeField] float zoom_speed = 50f;
    [SerializeField] float scroll_speed = 5.0f;

    bool track_mouse = false;

    [SerializeField] float zoom = 10.0f;
    [SerializeField] Vector2 zoomlimit = new Vector2(2f, 100f);

    void OnEnable() {
        transform = GetComponent<Transform>();
        camera = GetComponent<Camera>();
        instance = this;

        zoomlimit = Vector2.one;
        zoom = 1f;
    }

    void Update() {
        Movement();
        MouseMovement();
        ApplyZoom();
        track_mouse = Application.isFocused;
    }

    public static void AdjustToNetwork(NeuralNetwork network) {
        float height = 0;
        float avg = 0;
        foreach (Layer layer in network.Layers) {
            height = Math.Max(height, layer.NeuronNum);
            avg += layer.NeuronNum;
        }
        float width = network.Layers.Length - 1;
        avg /= network.Layers.Length;

        float mul = Visualization.instance.spacing / 2f;
        height *= mul;
        width *= mul;

        instance.zoomlimit.x = Math.Max(height/100f, 1f);
        instance.zoomlimit.y = height;
        instance.zoom = avg;

        instance.transform.position = new Vector3(width, 0, -10f);
    }

    public static bool MouseOnScreen() {
        Vector2 mouse_pos = Input.mousePosition;
        Vector2 screen_size = Visualization.instance.GetComponent<RectTransform>().sizeDelta;
        bool x = 0 <= mouse_pos.x && mouse_pos.x <= screen_size.x;
        bool y = 0 <= mouse_pos.y && mouse_pos.y <= screen_size.y;
        return x && y;
    }

    void Movement() {
        Vector3 move = Vector3.zero;

        if (Input.GetKey(KeyCode.W)) { move.y += 1f; }
        if (Input.GetKey(KeyCode.S)) { move.y -= 1f; }
        if (Input.GetKey(KeyCode.D)) { move.x += 1f; }
        if (Input.GetKey(KeyCode.A)) { move.x -= 1f; }

        transform.position += move * move_speed * Time.deltaTime;
    }

    void MouseMovement() {
        if (!track_mouse) return;
        if (!MouseOnScreen()) return;
        if (!Input.GetMouseButton(0)) return;

        Vector2 change = mouse_speed * zoom / Screen.height * Input.mousePositionDelta;

        transform.position -= (Vector3)change;
    }

    void ApplyZoom() {
        float z = 0f;
        if (Input.GetKey(KeyCode.E)) { z += 1f; }
        if (Input.GetKey(KeyCode.Q)) { z -= 1f; }
        if (z == 0f && !MouseOnScreen()) return;
        z *= zoom_speed * Time.deltaTime;

        z -= scroll_speed * Input.mouseScrollDelta.y;

        if (Input.GetKey(KeyCode.LeftControl)) {
            Visualization.ChangeSpace(z / 5f);
        } else {
            Vector3 preLoc = camera.ScreenToWorldPoint(Input.mousePosition);

            zoom = Mathf.Clamp(zoom + z, zoomlimit.x, zoomlimit.y);
            camera.orthographicSize = zoom;

            transform.position += preLoc - camera.ScreenToWorldPoint(Input.mousePosition);
        }
    }
}
