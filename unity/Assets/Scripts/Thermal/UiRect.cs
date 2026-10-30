using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// A rounded rectangle (P2Pro/UIRect) for the panel and the HUD: a quad of the given size in
    /// metres, coloured through a MaterialPropertyBlock, so one material serves every rectangle
    /// of a layer. Optionally keeps the quad's collider for the controller rays.
    /// </summary>
    public class UiRect : MonoBehaviour
    {
        static readonly int ColourId = Shader.PropertyToID("_Colour");
        static readonly int BorderColourId = Shader.PropertyToID("_BorderColour");
        static readonly int SizeId = Shader.PropertyToID("_Size");
        static readonly int RadiusId = Shader.PropertyToID("_Radius");
        static readonly int BorderWidthId = Shader.PropertyToID("_BorderWidth");
        static readonly int TexColourId = Shader.PropertyToID("_TexColour");
        static readonly int MainTexId = Shader.PropertyToID("_MainTex");
        static readonly Dictionary<bool, Material> Materials = new();

        MeshRenderer _renderer;
        MaterialPropertyBlock _block;
        Vector2 _size;
        Color _colour, _borderColour;
        float _radius, _border;

        public Collider Collider { get; private set; }
        public Vector2 Size => _size;

        /// <summary>Layer among rectangles of the same material: higher draws over lower.</summary>
        public int Order { set => _renderer.sortingOrder = value; }

        /// <summary><paramref name="onTop"/>: drawn over everything (the HUD).</summary>
        public static UiRect Create(Transform parent, string name, Vector3 localPosition, Vector2 size, Color colour,
            float radius = 0.006f, bool onTop = false, bool collider = false)
        {
            var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            quad.name = name;
            if (!collider) Destroy(quad.GetComponent<Collider>());
            quad.transform.SetParent(parent, false);
            quad.transform.localPosition = localPosition;
            var rect = quad.AddComponent<UiRect>();
            rect.Collider = collider ? quad.GetComponent<Collider>() : null;
            rect._renderer = quad.GetComponent<MeshRenderer>();
            rect._renderer.sharedMaterial = SharedMaterial(onTop);
            rect._renderer.shadowCastingMode = ShadowCastingMode.Off;
            rect._block = new MaterialPropertyBlock();
            rect._radius = radius;
            rect._colour = colour;
            rect._borderColour = Color.clear;
            rect.SetSize(size);
            return rect;
        }

        static Material SharedMaterial(bool onTop)
        {
            if (Materials.TryGetValue(onTop, out var material) && material != null) return material;
            material = new Material(Resources.Load<Shader>("P2ProUIRect"));
            if (onTop)
            {
                material.SetFloat("_ZTest", (float)CompareFunction.Always);
                material.renderQueue = 3999;        // under the HUD's text (4000)
            }
            Materials[onTop] = material;
            return material;
        }

        public void SetSize(Vector2 size)
        {
            _size = size;
            transform.localScale = new Vector3(size.x, size.y, 1);
            Apply();
        }

        public Color Colour
        {
            get => _colour;
            set { if (_colour != value) { _colour = value; Apply(); } }
        }

        public void SetBorder(Color colour, float width)
        {
            if (_borderColour == colour && Mathf.Approximately(_border, width)) return;
            _borderColour = colour;
            _border = width;
            Apply();
        }

        /// <summary>Fills from a texture (the palette), keeping the colour's alpha.</summary>
        public void SetTexture(Texture texture)
        {
            if (texture == null) return;
            _block.SetTexture(MainTexId, texture);
            _block.SetFloat(TexColourId, 1);
            _renderer.SetPropertyBlock(_block);
        }

        void Apply()
        {
            _block.SetColor(ColourId, _colour);
            _block.SetColor(BorderColourId, _borderColour);
            _block.SetVector(SizeId, _size);
            _block.SetFloat(RadiusId, _radius);
            _block.SetFloat(BorderWidthId, _border);
            _renderer.SetPropertyBlock(_block);
        }
    }
}
