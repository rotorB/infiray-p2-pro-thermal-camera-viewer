package com.p2pro.quest

import java.nio.ByteBuffer

/**
 * Byte-exact UVC frames from the camera, through libuvc directly (native/raw_uvc.c).
 *
 * UVCAndroid's frame callback round-trips YUYV through RGBX even in PIXEL_FORMAT_RAW, which
 * corrupts the P2 Pro's 16-bit temperature half. One camera at a time.
 */
object RawUvc {
    init {
        System.loadLibrary("p2prouvc")
    }

    /** Starts YUYV streaming on the device behind [fd]. Returns 0 or a libuvc error code. */
    fun open(fd: Int, width: Int, height: Int, fps: Int): Int = nativeOpen(fd, width, height, fps)

    /**
     * Blocks until a frame newer than the last one returned arrives and copies it into [buffer]
     * (direct, at least width*height*2 bytes). Returns its size, 0 on timeout, -1 once closed.
     */
    fun waitFrame(buffer: ByteBuffer, timeoutMs: Int): Int = nativeWaitFrame(buffer, timeoutMs)

    /** Stops streaming and wakes any waiting reader. Safe to call when nothing is open. */
    fun close() = nativeClose()

    private external fun nativeOpen(fd: Int, width: Int, height: Int, fps: Int): Int
    private external fun nativeWaitFrame(buffer: ByteBuffer, timeoutMs: Int): Int
    private external fun nativeClose()
}
