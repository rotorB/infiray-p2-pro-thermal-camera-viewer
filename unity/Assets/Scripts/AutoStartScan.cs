using Genesis.RoomScan;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Starts a live room scan on launch, shown as a wireframe. QuestRoomScan's
    /// game-ready preset leaves starting the scan to the host app; StartScanAsync
    /// also asks for the USE_SCENE, HEADSET_CAMERA and USE_ANCHOR_API permissions.
    /// </summary>
    public class AutoStartScan : MonoBehaviour
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Install()
        {
            if (RoomScanSession.Instance == null)
            {
                Debug.LogWarning("[P2Pro] No RoomScanSession in the scene, scan not started");
                return;
            }
            new GameObject(nameof(AutoStartScan)).AddComponent<AutoStartScan>();
        }

        async void Start()
        {
            // RoomScanner initialises its modules in its own Start: let it run first.
            await Awaitable.NextFrameAsync();
            Debug.Log("[P2Pro] Starting room scan");
            await RoomScanSession.Instance.StartScanAsync();
            Debug.Log("[P2Pro] Room scan running: " + RoomScanSession.Instance.IsScanning);
            if (RoomScanner.Instance != null)
                RoomScanner.Instance.SetRenderMode(ScanRenderMode.Wireframe);

            // After the scan's permission dialogs: a second request while one is showing is
            // dropped by Android without UI.
            gameObject.AddComponent<ThermalPanel>();
        }

        void Update()
        {
            // X on the left controller cycles Wireframe -> Vertex -> ... (unavailable modes are skipped).
            if (OVRInput.GetDown(OVRInput.Button.Three) && RoomScanner.Instance != null)
                RoomScanner.Instance.CycleRenderMode();
        }
    }
}
