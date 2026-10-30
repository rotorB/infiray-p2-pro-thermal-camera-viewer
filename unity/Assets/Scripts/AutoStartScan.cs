using Genesis.RoomScan;
using Meta.XR.MRUtilityKit;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Starts a live room scan on launch, shown as a wireframe. QuestRoomScan's
    /// game-ready preset leaves starting the scan to the host app; StartScanAsync
    /// also asks for the USE_SCENE, HEADSET_CAMERA and USE_ANCHOR_API permissions.
    /// The scan volume is a fixed box around Unity's world origin (19.2 x 6.4 x 19.2 m, see
    /// VolumeIntegrator.voxelCount), and the world origin is wherever the headset's tracking
    /// started, possibly another room. So before scanning, the world is shifted to put the head
    /// at the volume's centre horizontally, and the floor <see cref="VolumeCentreHeight"/>
    /// below it, so the walls fit even with small voxels (ScanTuning).
    /// Quest's depth service sometimes refuses the app's depth provider (system log
    /// "EnvironmentDepth ... Failed to create layer"), and Unity starts it only once; while no
    /// depth arrives the capture is restarted every few seconds. The boundary is suppressed.
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

        /// <summary>Height of the scan volume's centre above the floor, m.</summary>
        public const float VolumeCentreHeight = 1.3f;
        const float TrackingWaitSeconds = 5;
        const float DepthRetrySeconds = 6;

        bool _scanning;
        float _depthSeenAt;
        int _depthRetries;

        async void Start()
        {
            // RoomScanner initialises its modules in its own Start: let it run first.
            await Awaitable.NextFrameAsync();
            await CentreWorldOnHead();
            Debug.Log("[P2Pro] Starting room scan");
            await RoomScanSession.Instance.StartScanAsync();
            Debug.Log("[P2Pro] Room scan running: " + RoomScanSession.Instance.IsScanning);
            _scanning = RoomScanSession.Instance.IsScanning;
            _depthSeenAt = Time.realtimeSinceStartup;
            // QuestRoomScan's own mesh (RGB colours, white grid) stays hidden: only the thermal
            // overlays are drawn on it (they draw in every render mode).
            if (RoomScanner.Instance != null)
                RoomScanner.Instance.SetRenderMode(ScanRenderMode.None);
            GPUMeshRenderer.OverlayMaterials.Insert(0, new Material(Resources.Load<Shader>("P2ProMeshDepth")));

            // After the scan's permission dialogs: a second request while one is showing is
            // dropped by Android without UI.
            gameObject.AddComponent<ThermalPanel>();
        }

        void WatchDepth()
        {
            var depth = DepthCapture.Instance;
            if (!_scanning || depth == null) return;
            if (DepthCapture.DepthAvailable)
            {
                if (_depthRetries > 0) Debug.Log($"[P2Pro] depth is back after {_depthRetries} restarts");
                _depthRetries = 0;
                _depthSeenAt = Time.realtimeSinceStartup;
                return;
            }
            if (Time.realtimeSinceStartup - _depthSeenAt < DepthRetrySeconds) return;
            _depthSeenAt = Time.realtimeSinceStartup;
            _depthRetries++;
            Debug.LogWarning($"[P2Pro] no depth for {DepthRetrySeconds} s, restarting depth capture ({_depthRetries})");
            depth.StopDepthCapture();
            depth.StartDepthCapture();
        }

        // Moves the tracking space so the head is at x = z = 0 and the floor (the tracking
        // origin is floor level) at y = -VolumeCentreHeight. MRUK's world lock, when active,
        // owns the TrackingSpace pose and applies TrackingSpaceOffset; otherwise the rig root
        // moves it. The offset is set in both cases in case world lock starts later.
        static async Awaitable CentreWorldOnHead()
        {
            float deadline = Time.realtimeSinceStartup + TrackingWaitSeconds;
            while (Time.realtimeSinceStartup < deadline &&
                   (!OVRPlugin.GetNodePositionTracked(OVRPlugin.Node.Head) ||
                    (MRUK.Instance != null && !MRUK.Instance.IsInitialized)))
                await Awaitable.NextFrameAsync();

            var rig = FindFirstObjectByType<OVRCameraRig>();
            if (rig == null || rig.centerEyeAnchor == null)
            {
                Debug.LogWarning("[P2Pro] No OVRCameraRig, world not centred on the head");
                return;
            }
            var head = rig.centerEyeAnchor.position;
            var shift = new Vector3(-head.x, -VolumeCentreHeight, -head.z);
            var mruk = MRUK.Instance;
            bool worldLock = mruk != null && mruk.IsWorldLockActive;
            if (mruk != null) mruk.TrackingSpaceOffset = Matrix4x4.Translate(shift) * mruk.TrackingSpaceOffset;
            if (!worldLock) rig.transform.position += shift;
            Debug.Log($"[P2Pro] head at {head:F2} m, world shifted by {shift:F2} m (world lock {worldLock})");
        }

        void Update()
        {
            if (OVRManager.instance != null) OVRManager.instance.shouldBoundaryVisibilityBeSuppressed = true;
            WatchDepth();
        }
    }
}
