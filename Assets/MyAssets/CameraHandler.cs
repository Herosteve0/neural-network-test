using UnityEngine;

public class CameraHandler : MonoBehaviour
{
    Transform transform;
    Camera camera;

    [SerializeField] float move_speed = 25.0f;
    [SerializeField] float mouse_speed = 0.5f;
    [SerializeField] float zoom_speed = 50f;
    [SerializeField] float scroll_speed = 5.0f;

    float zoom = 10.0f;
    Vector2 zoomlimit = new Vector2(0.5f, 100f);

    void Awake() {
        transform = GetComponent<Transform>();
        camera = GetComponent<Camera>();
    }

    void Update() {
        Movement();
        MouseMovement();
        ApplyZoom();
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
        if (!Input.GetMouseButton(0)) {
            return;
        }
        transform.position -= zoom / zoomlimit.y * mouse_speed * Input.mousePositionDelta;
    }

    void ApplyZoom() {
        float z = 0f;
        if (Input.GetKey(KeyCode.E)) { z += 1f; }
        if (Input.GetKey(KeyCode.Q)) { z -= 1f; }
        z *= zoom_speed * Time.deltaTime;

        z -= scroll_speed * Input.mouseScrollDelta.y;

        zoom = Mathf.Min(Mathf.Max(zoom+z, zoomlimit[0]), zoomlimit[1]);
        camera.orthographicSize = zoom;
    }
}
