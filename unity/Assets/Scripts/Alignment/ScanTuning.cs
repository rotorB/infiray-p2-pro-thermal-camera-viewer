using System;
using Genesis.RoomScan;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Room scan geometry settings from the Setup tab, remembered:
    /// <list type="bullet">
    /// <item>Voxel size, 2.5 to 8 cm. The voxel grid keeps its count (384 x 128 x 384), so smaller
    /// voxels mean finer geometry in a smaller volume around the world origin (the head at
    /// start, the floor 1.3 m below the volume's centre: <see cref="AutoStartScan"/>). Changing
    /// it starts the scan and the painted room over; it is applied a second after the slider
    /// stops.</item>
    /// <item>Freeze seen surfaces: once a second, voxels in view that reached nearly the full
    /// weight (seen from enough frames) are frozen: they stop integrating, so the mesh there
    /// stops being rebuilt. The rest keeps scanning. Unfreeze all lets them refine again.</item>
    /// </list>
    /// </summary>
    public class ScanTuning : MonoBehaviour
    {
        public const float MinVoxelCm = 2.5f, MaxVoxelCm = 8;
        const float VoxelStepCm = 0.5f;
        const float ApplyDelay = 1;
        const float FreezeInterval = 1;
        const float FreezeConeDeg = 35;
        const float FreezeRange = 4;            // m
        const float FreezeWeightShare = 0.9f;   // of the weight a voxel converges to
        const string Key = "p2pro.scan.";

        float _voxelCm, _appliedCm;
        float _applyAt = -1;
        float _nextFreeze;
        Transform _head;

        /// <summary>The volume was re-allocated: the scan and everything placed in it start over.</summary>
        public event Action VolumeChanged;

        public float VoxelCm => _voxelCm;
        public bool FreezeSeen { get; private set; }
        public string FreezeLabel => $"Freeze seen surfaces: {(FreezeSeen ? "on" : "off")}";

        public string VoxelLabel(float cm)
        {
            var volume = VolumeIntegrator.Instance;
            if (volume == null) return $"Voxel {cm:F1} cm";
            float size = cm / 100;
            string room = $"room up to {volume.VoxelCount.x * size:F1} m wide, {AutoStartScan.VolumeCentreHeight + volume.VoxelCount.y * size / 2:F1} m high";
            return $"Voxel {cm:F1} cm: {room}" + (_applyAt >= 0 ? " (applying)" : "");
        }

        void Start()
        {
            FreezeSeen = PlayerPrefs.GetInt(Key + "freeze", 1) == 1;
            _voxelCm = Mathf.Clamp(PlayerPrefs.GetFloat(Key + "voxel", 5), MinVoxelCm, MaxVoxelCm);
            _appliedCm = VolumeIntegrator.Instance != null ? VolumeIntegrator.Instance.VoxelSize * 100 : 5;
            if (!Mathf.Approximately(_voxelCm, _appliedCm)) _applyAt = Time.unscaledTime;
        }

        /// <summary>Slider: snapped to 0.5 cm, applied once it rests.</summary>
        public void SetVoxelCm(float cm)
        {
            cm = Mathf.Clamp(Mathf.Round(cm / VoxelStepCm) * VoxelStepCm, MinVoxelCm, MaxVoxelCm);
            if (Mathf.Approximately(cm, _voxelCm)) return;
            _voxelCm = cm;
            _applyAt = Time.unscaledTime + ApplyDelay;
        }

        public void ToggleFreeze()
        {
            FreezeSeen = !FreezeSeen;
            PlayerPrefs.SetInt(Key + "freeze", FreezeSeen ? 1 : 0);
            Debug.Log("[P2Pro] " + FreezeLabel);
        }

        public void UnfreezeAll()
        {
            VolumeIntegrator.Instance?.UnfreezeAll();
            Debug.Log("[P2Pro] scan unfrozen");
        }

        void Update()
        {
            var volume = VolumeIntegrator.Instance;
            if (volume == null) return;

            if (_applyAt >= 0 && Time.unscaledTime >= _applyAt)
            {
                _applyAt = -1;
                if (!Mathf.Approximately(_voxelCm, _appliedCm))
                {
                    volume.SetVoxelSize(_voxelCm / 100);
                    _appliedCm = _voxelCm;
                    PlayerPrefs.SetFloat(Key + "voxel", _voxelCm);
                    PlayerPrefs.Save();
                    Debug.Log("[P2Pro] " + VoxelLabel(_voxelCm));
                    VolumeChanged?.Invoke();
                }
            }

            if (!FreezeSeen || volume.VolumesReleased || Time.unscaledTime < _nextFreeze || !TryGetHead()) return;
            _nextFreeze = Time.unscaledTime + FreezeInterval;
            volume.FreezeInView(_head.position, _head.forward, FreezeConeDeg, FreezeRange, volume.MaxWeight * FreezeWeightShare);
        }

        bool TryGetHead()
        {
            if (_head == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig != null) _head = rig.centerEyeAnchor;
            }
            return _head != null;
        }

        void OnDestroy() => PlayerPrefs.Save();
    }
}
