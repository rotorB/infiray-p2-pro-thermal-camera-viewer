package com.p2pro.quest

import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decodes raw P2 Pro frames the same way as app.py on the desktop.
 *
 * The camera sends YUYV 256x384: the top 192 rows are the camera's own 8-bit image, the
 * bottom 192 rows carry one little-endian uint16 per pixel in 1/64 K. Celsius = raw/64 - 273.15.
 */
class ThermalProcessor(private val onFrameReady: () -> Unit) {

    enum class RangeMode(val label: String, val fixedMin: Float, val fixedMax: Float) {
        AUTO("Auto", Float.NaN, Float.NaN),
        ROOM("Room 15–35 °C", 15f, 35f),
        WIDE("Wide −20–150 °C", -20f, 150f),
    }

    class Frame {
        val celsius = FloatArray(PIXELS)
        val argb = IntArray(PIXELS)
        var minC = 0f
        var maxC = 0f
        var minIndex = 0
        var maxIndex = 0
        var rangeMin = 0f
        var rangeMax = 1f
        var fps = 0f

        fun copyFrom(o: Frame) {
            o.celsius.copyInto(celsius)
            o.argb.copyInto(argb)
            minC = o.minC; maxC = o.maxC
            minIndex = o.minIndex; maxIndex = o.maxIndex
            rangeMin = o.rangeMin; rangeMax = o.rangeMax
            fps = o.fps
        }
    }

    @Volatile var palette = 0
    @Volatile var rangeMode = RangeMode.AUTO

    private val raw = ShortArray(PIXELS)
    private val work = Frame()
    private val published = Frame()
    private val lock = Any()
    @Volatile var frameCount = 0L
        private set
    @Volatile var badFrames = 0L
        private set

    private var autoMin = Float.NaN
    private var autoMax = Float.NaN
    private var fpsWindowStart = System.nanoTime()
    private var fpsWindowFrames = 0
    private var fps = 0f

    /** Decodes one raw YUYV 256x384 frame; the buffer must hold the camera's bytes unchanged. */
    fun onFrame(frame: ByteBuffer) {
        if (frame.remaining() < FRAME_BYTES) {
            if (badFrames++ % 100 == 0L) Log.w(TAG, "short frame: ${frame.remaining()} of $FRAME_BYTES bytes")
            return
        }
        frame.order(ByteOrder.LITTLE_ENDIAN)
        val shorts = frame.asShortBuffer()
        shorts.position(THERMAL_OFFSET_SHORTS)
        shorts.get(raw, 0, PIXELS)
        decode()
        synchronized(lock) { published.copyFrom(work) }
        if (frameCount++ % 250 == 0L) {
            Log.i(TAG, "frame $frameCount: %.1f..%.1f °C, %.0f fps".format(work.minC, work.maxC, work.fps))
        }
        onFrameReady()
    }

    /** Runs [block] on the latest decoded frame while holding the publish lock. */
    fun <T> withLatest(block: (Frame) -> T): T = synchronized(lock) { block(published) }

    private fun decode() {
        val f = work
        var lo = Float.MAX_VALUE
        var hi = -Float.MAX_VALUE
        var loI = 0
        var hiI = 0
        for (i in 0 until PIXELS) {
            val c = (raw[i].toInt() and 0xFFFF) / 64f - 273.15f
            f.celsius[i] = c
            if (c < lo) { lo = c; loI = i }
            if (c > hi) { hi = c; hiI = i }
        }
        f.minC = lo; f.maxC = hi; f.minIndex = loI; f.maxIndex = hiI

        val mode = rangeMode
        if (mode == RangeMode.AUTO) {
            // Smooth the endpoints so the palette does not flicker with single hot pixels.
            autoMin = if (autoMin.isNaN()) lo else autoMin + 0.2f * (lo - autoMin)
            autoMax = if (autoMax.isNaN()) hi else autoMax + 0.2f * (hi - autoMax)
            f.rangeMin = autoMin
            f.rangeMax = maxOf(autoMax, autoMin + 0.5f)
        } else {
            autoMin = Float.NaN; autoMax = Float.NaN
            f.rangeMin = mode.fixedMin
            f.rangeMax = mode.fixedMax
        }

        val lut = Palettes.all[palette].lut
        val scale = 255f / (f.rangeMax - f.rangeMin)
        for (i in 0 until PIXELS) {
            val idx = ((f.celsius[i] - f.rangeMin) * scale).toInt().coerceIn(0, 255)
            f.argb[i] = lut[idx]
        }

        fpsWindowFrames++
        val now = System.nanoTime()
        val dt = (now - fpsWindowStart) / 1e9f
        if (dt >= 1f) {
            fps = fpsWindowFrames / dt
            fpsWindowFrames = 0
            fpsWindowStart = now
        }
        f.fps = fps
    }

    companion object {
        private const val TAG = "P2ProQuest"
        const val WIDTH = 256
        const val HEIGHT = 192
        const val PIXELS = WIDTH * HEIGHT
        const val FRAME_W = 256
        const val FRAME_H = 384
        const val FRAME_BYTES = FRAME_W * FRAME_H * 2
        const val THERMAL_OFFSET_SHORTS = FRAME_W * HEIGHT
    }
}
