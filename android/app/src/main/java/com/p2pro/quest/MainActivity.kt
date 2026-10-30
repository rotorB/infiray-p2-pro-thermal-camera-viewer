package com.p2pro.quest

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.usb.UsbDevice
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import com.serenegiant.usb.USBMonitor
import java.nio.ByteBuffer
import kotlin.concurrent.thread

class MainActivity : Activity(), USBMonitor.OnDeviceConnectListener {

    private lateinit var view: ThermalView
    private lateinit var processor: ThermalProcessor
    private lateinit var usbMonitor: USBMonitor

    private val cameraLock = Any()
    private var reader: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        processor = ThermalProcessor { view.postInvalidate() }
        view = ThermalView(this, processor)
        setContentView(FrameLayout(this).apply {
            addView(view)
            addView(controls(), FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.START,
            ).apply { setMargins(16, 16, 16, 16) })
        })

        usbMonitor = USBMonitor(this, this)
    }

    private fun controls() = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        setBackgroundColor(Color.argb(140, 0, 0, 0))
        fun button(label: String, onClick: () -> Unit) = addView(Button(context).apply {
            text = label
            isAllCaps = false
            setOnClickListener { onClick() }
        })
        button("Palette") { processor.palette = (processor.palette + 1) % Palettes.all.size }
        button("Range") {
            val modes = ThermalProcessor.RangeMode.entries
            processor.rangeMode = modes[(processor.rangeMode.ordinal + 1) % modes.size]
        }
        button("Rotate") { view.quarterTurns = view.quarterTurns + 1 }
        button("Mirror") { view.mirrored = !view.mirrored }
    }

    override fun onStart() {
        super.onStart()
        val missing = REQUIRED_PERMISSIONS.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) {
            usbMonitor.register()
        } else {
            view.status = "Allow camera access to read the USB camera"
            requestPermissions(missing.toTypedArray(), REQUEST_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_CAMERA) return
        if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            view.status = "Waiting for P2 Pro…"
            usbMonitor.register()
        } else {
            view.status = "Camera permission denied"
        }
    }

    override fun onStop() {
        usbMonitor.unregister()
        closeCamera()
        super.onStop()
    }

    override fun onDestroy() {
        usbMonitor.destroy()
        super.onDestroy()
    }

    // --- USBMonitor callbacks (run on the monitor's worker thread) ---

    override fun onAttach(device: UsbDevice) {
        if (!isP2Pro(device)) return
        view.status = "P2 Pro found, requesting USB access…"
        usbMonitor.requestPermission(device)
    }

    override fun onDeviceOpen(device: UsbDevice, ctrlBlock: USBMonitor.UsbControlBlock, createNew: Boolean) {
        if (!isP2Pro(device)) return
        openCamera(ctrlBlock)
    }

    override fun onDeviceClose(device: UsbDevice, ctrlBlock: USBMonitor.UsbControlBlock) {
        closeCamera()
    }

    override fun onDetach(device: UsbDevice) {
        if (!isP2Pro(device)) return
        closeCamera()
        view.status = "P2 Pro disconnected"
    }

    override fun onCancel(device: UsbDevice) {
        view.status = "USB access was denied"
    }

    /**
     * Streams through [RawUvc], not UVCCamera: UVCAndroid converts every frame YUYV -> RGBX ->
     * YUYV before its frame callback, which scrambles the 16-bit temperature half.
     */
    private fun openCamera(ctrlBlock: USBMonitor.UsbControlBlock): Unit = synchronized(cameraLock) {
        if (reader != null) return
        detachKernelDriver(ctrlBlock)
        val rc = RawUvc.open(
            ctrlBlock.fileDescriptor, ThermalProcessor.FRAME_W, ThermalProcessor.FRAME_H, FPS,
        )
        if (rc != 0) {
            Log.e(TAG, "open failed: $rc")
            view.status = "Could not open P2 Pro (error $rc)"
            return
        }
        view.status = "Starting stream…"
        reader = thread(name = "uvc-reader") {
            val buffer = ByteBuffer.allocateDirect(ThermalProcessor.FRAME_BYTES)
            while (true) {
                val n = RawUvc.waitFrame(buffer, 1000)
                if (n < 0) break
                if (n == 0) continue
                buffer.clear().limit(n)
                processor.onFrame(buffer)
            }
        }
    }

    /**
     * Horizon OS binds the kernel uvcvideo driver to the P2 Pro (/dev/video*, not readable by
     * apps), so libuvc's own claim fails with BUSY. A forced claim from Java disconnects the
     * kernel driver; releasing right away leaves the interfaces free for libuvc.
     */
    private fun detachKernelDriver(ctrlBlock: USBMonitor.UsbControlBlock) {
        val device = ctrlBlock.device
        val connection = ctrlBlock.connection ?: return
        for (i in 0 until device.interfaceCount) {
            val intf = device.getInterface(i)
            if (connection.claimInterface(intf, true)) {
                connection.releaseInterface(intf)
            } else {
                Log.w(TAG, "could not detach kernel driver from interface ${intf.id}")
            }
        }
    }

    private fun closeCamera(): Unit = synchronized(cameraLock) {
        RawUvc.close()
        reader?.join()
        reader = null
    }

    private fun isP2Pro(d: UsbDevice) = d.vendorId == P2PRO_VID && d.productId == P2PRO_PID

    companion object {
        private const val TAG = "P2ProQuest"
        private const val REQUEST_CAMERA = 1
        // Horizon OS checks its own USB_CAMERA permission for UVC devices on top of CAMERA.
        private val REQUIRED_PERMISSIONS = listOf(Manifest.permission.CAMERA, "horizonos.permission.USB_CAMERA")
        const val P2PRO_VID = 0x0BDA
        const val P2PRO_PID = 0x5830
        private const val FPS = 25
    }
}
