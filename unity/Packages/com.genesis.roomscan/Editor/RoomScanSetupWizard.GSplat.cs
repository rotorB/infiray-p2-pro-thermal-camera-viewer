#if HAS_GAUSSIAN_SPLATTING
using GaussianSplatting.Runtime;
using Genesis.RoomScan.GSplat;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace Genesis.RoomScan.Editor
{
    public partial class RoomScanSetupWizard
    {
        const string GSPLAT_PKG = "Packages/com.genesis.roomscan/Runtime/GSplat/Shaders/";

        GSplatManager _gsplatManager;
        GaussianSplatRenderer _ugsRenderer;
        GSplatServerClient _gsplatServerClient;

        bool _ugsRendererWired;
        bool _ugsRenderFeatureAdded;
        bool _deferredRendering;

        bool _serverUrlCurrent;
        string _detectedLanIp;
        string _currentServerUrl;

        // -----------------------------------------------------------------
        //  Partial method implementations
        // -----------------------------------------------------------------

        partial void RefreshGSplat()
        {
            _gsplatManager = FindAny<GSplatManager>();
            _ugsRenderer = FindAny<GaussianSplatRenderer>();
            _gsplatServerClient = FindAny<GSplatServerClient>();

            _ugsRendererWired = _ugsRenderer != null && AreFieldsAssigned(_ugsRenderer,
                "m_ShaderSplats", "m_ShaderComposite", "m_ShaderDebugPoints", "m_ShaderDebugBoxes", "m_CSSplatUtilities");
            _ugsRenderFeatureAdded = HasUGSRenderFeature();
            _deferredRendering = IsDeferredRendering();
            RefreshServerUrl();
        }

        partial void DrawGSplatOptionalStatus()
        {
            StatusRowOptional("GSplatManager (Gaussian Splat)", _gsplatManager != null);
            StatusRowOptional("GaussianSplatRenderer (UGS)", _ugsRenderer != null);
            StatusRowOptional("GSplatServerClient (PC training)", _gsplatServerClient != null);
            if (_gsplatServerClient != null)
            {
                if (_serverUrlCurrent)
                {
                    StatusRow($"  Server URL \u2192 {_detectedLanIp}:8420", true);
                }
                else
                {
                    string stale = string.IsNullOrEmpty(_currentServerUrl) ? "(empty)" : _currentServerUrl;
                    StatusRow($"  Server URL stale: {stale} (LAN: {_detectedLanIp})", false);
                    if (GUILayout.Button("Fix Server URL"))
                    {
                        ConfigureServerUrl();
                        Refresh();
                    }
                }
            }
        }

        partial void CheckGSplatAnyMissing(ref bool anyMissing)
        {
            if (_gsplatManager == null) anyMissing = true;
        }

        partial void DrawGSplatShaderStatus(ref bool needsFix)
        {
            if (_ugsRenderer != null) { StatusRow("UGS renderer shaders + compute", _ugsRendererWired);        needsFix |= !_ugsRendererWired; }
            if (_ugsRenderer != null) { StatusRow("UGS RenderFeature on URP Renderer", _ugsRenderFeatureAdded); needsFix |= !_ugsRenderFeatureAdded; }
            if (_ugsRenderer != null) { StatusRow("URP Deferred Rendering (req. by UGS)", _deferredRendering);  needsFix |= !_deferredRendering; }
        }

        partial void WireGSplatComponents()
        {
            WireComponent(_ugsRenderer);
        }

        static void WireGSplatComponent(Component component)
        {
            if (component is GaussianSplatRenderer ugsr)
            {
                const string UGS_PKG = "Packages/org.nesnausk.gaussian-splatting/Shaders/";
                var so = new SerializedObject(ugsr);
                AssignAsset<Shader>(so, "m_ShaderSplats", UGS_PKG + "RenderGaussianSplats.shader");
                AssignAsset<Shader>(so, "m_ShaderComposite", UGS_PKG + "GaussianComposite.shader");
                AssignAsset<Shader>(so, "m_ShaderDebugPoints", UGS_PKG + "GaussianDebugRenderPoints.shader");
                AssignAsset<Shader>(so, "m_ShaderDebugBoxes", UGS_PKG + "GaussianDebugRenderBoxes.shader");
                AssignCompute(so, "m_CSSplatUtilities", UGS_PKG + "SplatUtilities.compute");
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(ugsr);

                if (!HasUGSRenderFeature()) AddUGSRenderFeature();
                if (!IsDeferredRendering()) SetDeferredRendering();
            }
        }

        partial void SetupGSplatIfAvailable(GameObject root)
        {
            SetupGSplatModule(root);
        }

        // -----------------------------------------------------------------
        //  GSplat-specific methods
        // -----------------------------------------------------------------

        internal static void SetupGSplatModule(GameObject root)
        {
            System.Type gsplatManagerType = null;
            System.Type gsplatClientType = null;
            foreach (var asm in System.AppDomain.CurrentDomain.GetAssemblies())
            {
                gsplatManagerType ??= asm.GetType("Genesis.RoomScan.GSplat.GSplatManager");
                gsplatClientType ??= asm.GetType("Genesis.RoomScan.GSplat.GSplatServerClient");
                if (gsplatManagerType != null && gsplatClientType != null) break;
            }

            if (gsplatManagerType == null)
            {
                Debug.LogWarning("[RoomScan Setup] GSplatManager type not found — is the GaussianSplatting package installed?");
                return;
            }

            if (root.GetComponent(gsplatManagerType) == null)
                Undo.AddComponent(root, gsplatManagerType);
            if (gsplatClientType != null && root.GetComponent(gsplatClientType) == null)
                Undo.AddComponent(root, gsplatClientType);

            var splatChild = root.transform.Find("SplatRenderer");
            if (splatChild == null)
            {
                var go = new GameObject("SplatRenderer");
                go.transform.SetParent(root.transform, false);
                Undo.RegisterCreatedObjectUndo(go, "Create SplatRenderer");
                splatChild = go.transform;
            }

            var ugsType = typeof(GaussianSplatRenderer);
            var ugsr = splatChild.GetComponent(ugsType);
            if (ugsr == null)
                ugsr = Undo.AddComponent(splatChild.gameObject, ugsType);

            WireComponent(ugsr);
            ConfigureServerUrl();
        }

        static void ConfigureServerUrl()
        {
            var client = FindAny<GSplatServerClient>();
            if (client == null) return;

            string lanIp = GetLanIp();
            if (string.IsNullOrEmpty(lanIp)) return;

            string url = $"http://{lanIp}:8420";

            var so = new SerializedObject(client);
            var prop = so.FindProperty("serverUrl");
            if (prop != null && prop.stringValue != url)
            {
                prop.stringValue = url;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(client);
                Debug.Log($"[RoomScanWizard] Server URL set to {url} (detected LAN IP: {lanIp})");
            }
        }

        void RefreshServerUrl()
        {
            _serverUrlCurrent = false;
            _detectedLanIp = GetLanIp();
            _currentServerUrl = null;

            if (_gsplatServerClient == null || string.IsNullOrEmpty(_detectedLanIp)) return;

            var so = new SerializedObject(_gsplatServerClient);
            var prop = so.FindProperty("serverUrl");
            if (prop == null) return;

            _currentServerUrl = prop.stringValue;
            string expected = $"http://{_detectedLanIp}:8420";
            _serverUrlCurrent = _currentServerUrl == expected;
        }

        static Material GetOrCreateSplatMaterial()
        {
            const string pkgMatPath = "Packages/com.genesis.roomscan/Runtime/GSplat/Materials/SplatRender.mat";
            var pkgMat = AssetDatabase.LoadAssetAtPath<Material>(pkgMatPath);
            if (pkgMat != null) return pkgMat;

            Shader shader = Shader.Find("Genesis/SplatRender");
            if (shader == null)
            {
                Debug.LogWarning("[RoomScan Setup] Shader 'Genesis/SplatRender' not found");
                return null;
            }

            if (!AssetDatabase.IsValidFolder("Assets/RoomScan"))
                AssetDatabase.CreateFolder("Assets", "RoomScan");

            var mat = new Material(shader) { name = "SplatRender" };
            const string matPath = "Assets/RoomScan/SplatRender.mat";
            AssetDatabase.CreateAsset(mat, matPath);
            AssetDatabase.SaveAssets();
            return mat;
        }

        static UniversalRendererData FindActiveRendererData()
        {
            var pipeline = UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline
                as UniversalRenderPipelineAsset;
            if (pipeline == null) return null;

            var so = new SerializedObject(pipeline);
            var list = so.FindProperty("m_RendererDataList");
            if (list == null || list.arraySize == 0) return null;

            return list.GetArrayElementAtIndex(0).objectReferenceValue
                as UniversalRendererData;
        }

        static bool HasUGSRenderFeature()
        {
            var rd = FindActiveRendererData();
            if (rd == null) return false;
            foreach (var f in rd.rendererFeatures)
            {
                if (f != null && f.GetType().Name == "GaussianSplatURPFeature") return true;
            }
            return false;
        }

        const int RENDERING_MODE_DEFERRED = 1;

        static bool IsDeferredRendering()
        {
            var rd = FindActiveRendererData();
            if (rd == null) return false;
            var so = new SerializedObject(rd);
            var prop = so.FindProperty("m_RenderingMode");
            return prop != null && prop.intValue == RENDERING_MODE_DEFERRED;
        }

        static void SetDeferredRendering()
        {
            var rd = FindActiveRendererData();
            if (rd == null)
            {
                Debug.LogWarning("[RoomScan Setup] No active URP RendererData found");
                return;
            }

            var so = new SerializedObject(rd);
            var prop = so.FindProperty("m_RenderingMode");
            if (prop == null) return;

            if (prop.intValue != RENDERING_MODE_DEFERRED)
            {
                prop.intValue = RENDERING_MODE_DEFERRED;
                so.ApplyModifiedProperties();
                EditorUtility.SetDirty(rd);
                AssetDatabase.SaveAssets();
                Debug.Log("[RoomScan Setup] Switched URP Renderer to Deferred rendering (required by UGS)");
            }
        }

        static void AddUGSRenderFeature()
        {
            var rd = FindActiveRendererData();
            if (rd == null)
            {
                Debug.LogWarning("[RoomScan Setup] No active URP RendererData found");
                return;
            }

            System.Type featureType = null;
            foreach (var asm in System.AppDomain.CurrentDomain.GetAssemblies())
            {
                featureType = asm.GetType("GaussianSplatting.Runtime.GaussianSplatURPFeature");
                if (featureType != null) break;
            }
            if (featureType == null)
            {
                Debug.LogWarning("[RoomScan Setup] GaussianSplatURPFeature type not found — " +
                    "ensure UGS package has GS_ENABLE_URP defined and Unity 6+");
                return;
            }

            var feature = (ScriptableRendererFeature)CreateInstance(featureType);
            feature.name = "Gaussian Splat Renderer";
            feature.SetActive(true);

            Undo.RecordObject(rd, "Add UGS Render Feature");
            AssetDatabase.AddObjectToAsset(feature, rd);

            var so = new SerializedObject(rd);
            var features = so.FindProperty("m_RendererFeatures");
            features.arraySize++;
            features.GetArrayElementAtIndex(features.arraySize - 1).objectReferenceValue = feature;
            so.ApplyModifiedProperties();

            EditorUtility.SetDirty(rd);
            AssetDatabase.SaveAssets();
            Debug.Log("[RoomScan Setup] Added GaussianSplatURPFeature to URP Renderer");
        }
    }
}
#endif
