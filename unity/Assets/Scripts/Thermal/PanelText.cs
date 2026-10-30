using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Multi-line status text on the debug panel: a TextMesh with the built-in font, drawn with
    /// P2Pro/Text and sized to <see cref="LineHeight"/> metres per line.
    /// </summary>
    public class PanelText : MonoBehaviour
    {
        public float LineHeight = 0.013f;

        TextMesh _text;
        Material _material;
        Font _font;
        bool _scaled;

        public string Text
        {
            get => _text.text;
            set { if (_text.text != value) _text.text = value; }
        }

        public static PanelText Create(Transform parent, Vector3 localPosition, string name,
            TextAnchor anchor = TextAnchor.UpperLeft, float lineHeight = 0.013f)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPosition;
            var panelText = go.AddComponent<PanelText>();
            panelText.LineHeight = lineHeight;
            panelText.Init(anchor);
            return panelText;
        }

        void Init(TextAnchor anchor)
        {
            _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _material = new Material(Resources.Load<Shader>("P2ProText")) { mainTexture = _font.material.mainTexture };
            Font.textureRebuilt += OnFontRebuilt;
            _text = gameObject.AddComponent<TextMesh>();
            _text.font = _font;
            _text.fontSize = 48;
            _text.anchor = anchor;
            _text.alignment = anchor == TextAnchor.MiddleCenter ? TextAlignment.Center : TextAlignment.Left;
            _text.color = Color.white;
            GetComponent<MeshRenderer>().sharedMaterial = _material;
        }

        /// <summary>Draws over everything, for labels placed on surfaces.</summary>
        public void AlwaysOnTop()
        {
            _material.SetFloat("_ZTest", (float)UnityEngine.Rendering.CompareFunction.Always);
            _material.renderQueue = 4000;       // Overlay
        }

        void OnFontRebuilt(Font font)
        {
            if (font == _font) _material.mainTexture = font.material.mainTexture;
        }

        // TextMesh size depends on font metrics: measure once and scale to LineHeight per line.
        void LateUpdate()
        {
            if (_scaled || string.IsNullOrEmpty(_text.text)) return;
            var bounds = GetComponent<MeshRenderer>().bounds;
            int lines = _text.text.Split('\n').Length;
            if (bounds.size.y <= 0) return;
            float perLine = bounds.size.y / transform.lossyScale.y / lines;
            transform.localScale = Vector3.one * (LineHeight / perLine);
            _scaled = true;
        }

        void OnDestroy()
        {
            Font.textureRebuilt -= OnFontRebuilt;
            Destroy(_material);
        }
    }
}
