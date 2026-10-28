package com.p2pro.quest

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View
import com.p2pro.quest.ThermalProcessor.Companion.HEIGHT
import com.p2pro.quest.ThermalProcessor.Companion.WIDTH
import java.util.Locale

/**
 * Draws the latest thermal frame scaled to fit, with min/max markers, a center spot,
 * the temperature under the controller pointer, and a color bar.
 */
class ThermalView(context: Context, private val processor: ThermalProcessor) : View(context) {

    /** Clockwise rotation in quarter turns (0..3). */
    var quarterTurns = 0
        set(value) { field = value and 3; invalidate() }
    var mirrored = false
        set(value) { field = value; invalidate() }
    var status: String = "Waiting for P2 Pro…"
        set(value) { field = value; postInvalidate() }

    private val frame = ThermalProcessor.Frame()
    private val pixels = IntArray(ThermalProcessor.PIXELS)
    private val bitmap = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888)
    private val toView = Matrix()
    private val toImage = Matrix()
    private val imageRect = RectF()
    private var hasFrame = false
    private var pointer: FloatArray? = null

    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 30f
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }
    private val bigText = Paint(text).apply { textSize = 44f; isFakeBoldText = true }
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.WHITE
        setShadowLayer(3f, 0f, 0f, Color.BLACK)
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG)

    init {
        setBackgroundColor(Color.BLACK)
    }

    override fun onDraw(canvas: Canvas) {
        hasFrame = processor.frameCount > 0
        if (!hasFrame) {
            drawCentered(canvas, status)
            return
        }
        processor.withLatest { frame.copyFrom(it) }
        frame.argb.copyInto(pixels)
        bitmap.setPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT)

        val barWidth = 140f
        layoutImage(width - barWidth, height.toFloat())
        canvas.save()
        canvas.concat(toView)
        canvas.drawBitmap(bitmap, 0f, 0f, bitmapPaint)
        canvas.restore()

        drawMarker(canvas, frame.maxIndex, Color.rgb(255, 60, 60), frame.maxC)
        drawMarker(canvas, frame.minIndex, Color.rgb(80, 160, 255), frame.minC)
        drawSpot(canvas, WIDTH / 2f, HEIGHT / 2f, spotTemperature(WIDTH / 2, HEIGHT / 2), Color.WHITE)
        pointer?.let { p ->
            val img = floatArrayOf(p[0], p[1])
            toImage.mapPoints(img)
            val x = img[0].toInt(); val y = img[1].toInt()
            if (x in 0 until WIDTH && y in 0 until HEIGHT) {
                drawSpot(canvas, x + 0.5f, y + 0.5f, frame.celsius[y * WIDTH + x], Color.GREEN)
            }
        }
        drawColorBar(canvas, width - barWidth, barWidth)

        val palette = Palettes.all[processor.palette].name
        canvas.drawText(
            String.format(Locale.US, "%.0f fps · %s · %s", frame.fps, palette, processor.rangeMode.label),
            imageRect.left + 16f, imageRect.top + 40f, text,
        )
    }

    private fun layoutImage(areaW: Float, areaH: Float) {
        val rotated = quarterTurns % 2 == 1
        val srcW = if (rotated) HEIGHT.toFloat() else WIDTH.toFloat()
        val srcH = if (rotated) WIDTH.toFloat() else HEIGHT.toFloat()
        val scale = minOf(areaW / srcW, areaH / srcH)
        val dw = srcW * scale
        val dh = srcH * scale
        imageRect.set((areaW - dw) / 2, (areaH - dh) / 2, (areaW + dw) / 2, (areaH + dh) / 2)

        toView.reset()
        toView.postTranslate(-WIDTH / 2f, -HEIGHT / 2f)
        if (mirrored) toView.postScale(-1f, 1f)
        toView.postRotate(90f * quarterTurns)
        toView.postScale(scale, scale)
        toView.postTranslate(imageRect.centerX(), imageRect.centerY())
        toView.invert(toImage)
    }

    private fun spotTemperature(cx: Int, cy: Int): Float {
        var sum = 0f
        var n = 0
        for (y in cy - 1..cy + 1) for (x in cx - 1..cx + 1) {
            sum += frame.celsius[y * WIDTH + x]; n++
        }
        return sum / n
    }

    private fun drawMarker(canvas: Canvas, index: Int, color: Int, celsius: Float) {
        val p = floatArrayOf(index % WIDTH + 0.5f, index / WIDTH + 0.5f)
        toView.mapPoints(p)
        val path = Path().apply {
            moveTo(p[0], p[1])
            lineTo(p[0] - 14f, p[1] - 24f)
            lineTo(p[0] + 14f, p[1] - 24f)
            close()
        }
        fill.color = color
        canvas.drawPath(path, fill)
        canvas.drawText(formatC(celsius), p[0] + 18f, p[1] - 8f, text)
    }

    private fun drawSpot(canvas: Canvas, ix: Float, iy: Float, celsius: Float, color: Int) {
        val p = floatArrayOf(ix, iy)
        toView.mapPoints(p)
        stroke.color = color
        canvas.drawLine(p[0] - 22f, p[1], p[0] - 6f, p[1], stroke)
        canvas.drawLine(p[0] + 6f, p[1], p[0] + 22f, p[1], stroke)
        canvas.drawLine(p[0], p[1] - 22f, p[0], p[1] - 6f, stroke)
        canvas.drawLine(p[0], p[1] + 6f, p[0], p[1] + 22f, stroke)
        bigText.color = color
        canvas.drawText(formatC(celsius), p[0] + 26f, p[1] - 12f, bigText)
    }

    private fun drawColorBar(canvas: Canvas, left: Float, barWidth: Float) {
        val top = 70f
        val bottom = height - 70f
        val x0 = left + 24f
        val x1 = x0 + 32f
        val lut = Palettes.all[processor.palette].lut
        val steps = 128
        val h = (bottom - top) / steps
        for (i in 0 until steps) {
            fill.color = lut[255 - i * 255 / (steps - 1)]
            canvas.drawRect(x0, top + i * h, x1, top + (i + 1) * h + 1, fill)
        }
        canvas.drawText(formatC(frame.rangeMax), x0 - 8f, top - 16f, text)
        canvas.drawText(formatC(frame.rangeMin), x0 - 8f, bottom + 40f, text)
    }

    private fun drawCentered(canvas: Canvas, message: String) {
        val w = bigText.measureText(message)
        bigText.color = Color.WHITE
        canvas.drawText(message, (width - w) / 2, height / 2f, bigText)
    }

    private fun formatC(c: Float) = String.format(Locale.US, "%.1f°", c)

    // Quest controllers and hands send hover events while pointing and touch events on pinch.
    override fun onHoverEvent(event: MotionEvent): Boolean {
        updatePointer(event)
        return true
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        updatePointer(event)
        return true
    }

    private fun updatePointer(event: MotionEvent) {
        pointer = when (event.actionMasked) {
            MotionEvent.ACTION_HOVER_EXIT, MotionEvent.ACTION_CANCEL -> null
            else -> floatArrayOf(event.x, event.y)
        }
        invalidate()
    }
}
