package com.p2pro.quest

import kotlin.math.roundToInt

/** 256-entry ARGB lookup tables, index 0 = coldest. */
object Palettes {
    class Palette(val name: String, val lut: IntArray)

    // Declared before `all`: object properties initialize in order and ironbow() reads this.
    private val ironbowStops = arrayOf(
        0.00f to intArrayOf(0, 0, 0),
        0.15f to intArrayOf(30, 0, 100),
        0.35f to intArrayOf(130, 0, 150),
        0.55f to intArrayOf(220, 60, 40),
        0.75f to intArrayOf(255, 160, 0),
        0.90f to intArrayOf(255, 230, 80),
        1.00f to intArrayOf(255, 255, 255),
    )

    val all: List<Palette> = listOf(
        Palette("Inferno", build(::inferno)),
        Palette("Turbo", build(::turbo)),
        Palette("Ironbow", build(::ironbow)),
        Palette("White hot", build { t -> floatArrayOf(t, t, t) }),
        Palette("Black hot", build { t -> floatArrayOf(1 - t, 1 - t, 1 - t) }),
    )

    private fun build(f: (Float) -> FloatArray): IntArray = IntArray(256) { i ->
        val c = f(i / 255f)
        val r = (c[0].coerceIn(0f, 1f) * 255).roundToInt()
        val g = (c[1].coerceIn(0f, 1f) * 255).roundToInt()
        val b = (c[2].coerceIn(0f, 1f) * 255).roundToInt()
        (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }

    // Polynomial fit of matplotlib's inferno (Matt Zucker, CC0).
    private fun inferno(t: Float): FloatArray {
        val c = arrayOf(
            floatArrayOf(0.00021894037f, 0.0016510046f, -0.019480898f),
            floatArrayOf(0.10651342f, 0.56395644f, 3.9327124f),
            floatArrayOf(11.602493f, -3.972854f, -15.942394f),
            floatArrayOf(-41.703995f, 17.436399f, 44.354145f),
            floatArrayOf(77.16294f, -33.40236f, -81.80731f),
            floatArrayOf(-71.31943f, 32.626064f, 73.20952f),
            floatArrayOf(25.131126f, -12.242669f, -23.070325f),
        )
        return FloatArray(3) { ch ->
            var v = c[6][ch]
            for (k in 5 downTo 0) v = v * t + c[k][ch]
            v
        }
    }

    // Polynomial approximation of Google's Turbo (Apache 2.0).
    private fun turbo(t: Float): FloatArray {
        val r = 0.13572138f + t * (4.6153926f + t * (-42.66032f + t * (132.13108f + t * (-152.9424f + t * 59.28638f))))
        val g = 0.09140261f + t * (2.1941884f + t * (4.8429666f + t * (-14.185033f + t * (4.2772985f + t * 2.829566f))))
        val b = 0.1066733f + t * (12.641946f + t * (-60.582047f + t * (110.36277f + t * (-89.90311f + t * 27.34825f))))
        return floatArrayOf(r, g, b)
    }

    private fun ironbow(t: Float): FloatArray {
        for (i in 1 until ironbowStops.size) {
            val (t1, c1) = ironbowStops[i]
            if (t <= t1) {
                val (t0, c0) = ironbowStops[i - 1]
                val u = (t - t0) / (t1 - t0)
                return FloatArray(3) { ch -> (c0[ch] + (c1[ch] - c0[ch]) * u) / 255f }
            }
        }
        return floatArrayOf(1f, 1f, 1f)
    }
}
