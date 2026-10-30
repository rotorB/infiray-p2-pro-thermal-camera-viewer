using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.XR.OpenXR.Features;
using UnityEngine;

namespace P2Pro.Editor
{
    /// <summary>
    /// Command-line Quest build:
    /// Unity -batchmode -quit -projectPath unity -executeMethod P2Pro.Editor.QuestBuild.Build
    /// Writes Builds/p2pro.apk.
    /// </summary>
    public static class QuestBuild
    {
        const string ApkPath = "Builds/p2pro.apk";

        // Meta's AR Foundation features. QuestRoomScan's wizard only marks them
        // "recommended", but without Session + Occlusion the occlusion subsystem
        // is null and the scan receives no depth frames.
        static readonly string[] RequiredFeatures =
        {
            "com.unity.openxr.feature.arfoundation-meta-session",
            "com.unity.openxr.feature.arfoundation-meta-occlusion",
            "com.unity.openxr.feature.arfoundation-meta-camera",
        };

        [MenuItem("P2Pro/Configure OpenXR features")]
        public static void Configure()
        {
            FeatureHelpers.RefreshFeatures(BuildTargetGroup.Android);
            foreach (var id in RequiredFeatures)
            {
                var feature = FeatureHelpers.GetFeatureWithIdForBuildTarget(BuildTargetGroup.Android, id);
                if (feature == null)
                    throw new InvalidOperationException("OpenXR feature not found: " + id);
                if (feature.enabled) continue;
                feature.enabled = true;
                EditorUtility.SetDirty(feature);
                Debug.Log("[P2Pro] Enabled OpenXR feature " + id);
            }
            AssetDatabase.SaveAssets();
        }

        public static void Build()
        {
            Configure();

            var scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray();
            if (scenes.Length == 0)
                throw new InvalidOperationException("No enabled scenes in the build list");

            Directory.CreateDirectory(Path.GetDirectoryName(ApkPath));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = ApkPath,
                target = BuildTarget.Android,
                options = BuildOptions.None,
            });

            var summary = report.summary;
            Debug.Log($"[P2Pro] Build {summary.result}: {summary.outputPath}, " +
                      $"{summary.totalSize / (1024 * 1024)} MB, {summary.totalTime}, scenes: {string.Join(", ", scenes)}");
            if (summary.result != BuildResult.Succeeded)
                EditorApplication.Exit(1);
        }
    }
}
