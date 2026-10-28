package com.p2pro.quest.unity;

import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbInterface;
import android.hardware.usb.UsbManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

/**
 * Finds the P2 Pro, asks for USB access and opens it. Unity reads the file descriptor and
 * streams through libp2prouvc (p2pro_open / p2pro_take_celsius / p2pro_close).
 *
 * The runtime permissions CAMERA and horizonos.permission.USB_CAMERA must be granted before
 * {@link #start}: Android refuses access to USB video-class devices without CAMERA.
 *
 * Called from C# through AndroidJavaClass. All state changes run on the main thread; the
 * getters are safe from any thread.
 */
public final class UsbThermal {
    private static final String TAG = "P2ProThermal";
    private static final int P2PRO_VID = 0x0BDA;
    private static final int P2PRO_PID = 0x5830;
    private static final String ACTION_USB_PERMISSION = "com.p2pro.quest.unity.USB_PERMISSION";
    private static final long POLL_MS = 1000;

    private static final Handler main = new Handler(Looper.getMainLooper());
    private static Context context;
    private static UsbManager usb;
    private static UsbDeviceConnection connection;
    private static boolean permissionPending;

    private static volatile int fd = -1;
    private static volatile int generation;
    private static volatile String status = "Not started";

    private UsbThermal() {}

    /** Starts looking for the camera; requests USB access when it appears. */
    public static void start(Context ctx) {
        Context app = ctx.getApplicationContext();
        main.post(() -> {
            if (context != null) return;
            context = app;
            usb = (UsbManager) app.getSystemService(Context.USB_SERVICE);
            IntentFilter filter = new IntentFilter(ACTION_USB_PERMISSION);
            filter.addAction(UsbManager.ACTION_USB_DEVICE_DETACHED);
            if (Build.VERSION.SDK_INT >= 33) {
                app.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED);
            } else {
                app.registerReceiver(receiver, filter);
            }
            poll.run();
        });
    }

    /** Closes the device and stops looking for it. */
    public static void stop() {
        main.post(() -> {
            if (context == null) return;
            main.removeCallbacks(poll);
            context.unregisterReceiver(receiver);
            closeDevice("Stopped");
            context = null;
            permissionPending = false;
        });
    }

    /** USB file descriptor of the open camera, or -1. */
    public static int getFd() { return fd; }

    /** Bumped every time the camera is opened or closed, so Unity can reopen its stream. */
    public static int getGeneration() { return generation; }

    /** Human-readable state for the UI. */
    public static String getStatus() { return status; }

    // Hot-plug: Horizon OS does not deliver USB_DEVICE_ATTACHED to receivers registered at
    // runtime reliably, so poll the device list until the camera is open.
    private static final Runnable poll = new Runnable() {
        @Override public void run() {
            if (context == null) return;
            if (connection == null && !permissionPending) {
                UsbDevice device = findCamera();
                if (device == null) {
                    setStatus("Plug in the P2 Pro");
                } else if (usb.hasPermission(device)) {
                    openDevice(device);
                } else {
                    requestPermission(device);
                }
            }
            main.postDelayed(this, POLL_MS);
        }
    };

    private static final BroadcastReceiver receiver = new BroadcastReceiver() {
        @Override public void onReceive(Context ctx, Intent intent) {
            UsbDevice device = intent.getParcelableExtra(UsbManager.EXTRA_DEVICE);
            if (device == null || !isCamera(device)) return;
            if (ACTION_USB_PERMISSION.equals(intent.getAction())) {
                permissionPending = false;
                if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                    openDevice(device);
                } else {
                    setStatus("USB access to the P2 Pro was denied");
                }
            } else if (UsbManager.ACTION_USB_DEVICE_DETACHED.equals(intent.getAction())) {
                closeDevice("P2 Pro disconnected");
            }
        }
    };

    private static UsbDevice findCamera() {
        for (UsbDevice device : usb.getDeviceList().values()) {
            if (isCamera(device)) return device;
        }
        return null;
    }

    private static boolean isCamera(UsbDevice d) {
        return d.getVendorId() == P2PRO_VID && d.getProductId() == P2PRO_PID;
    }

    private static void requestPermission(UsbDevice device) {
        // Mutable so the system can add EXTRA_DEVICE and EXTRA_PERMISSION_GRANTED; explicit
        // (package set), which API 34+ requires for mutable PendingIntents.
        Intent intent = new Intent(ACTION_USB_PERMISSION).setPackage(context.getPackageName());
        int flags = Build.VERSION.SDK_INT >= 31 ? PendingIntent.FLAG_MUTABLE : 0;
        usb.requestPermission(device, PendingIntent.getBroadcast(context, 0, intent, flags));
        permissionPending = true;
        setStatus("Allow USB access to the P2 Pro");
    }

    private static void openDevice(UsbDevice device) {
        if (connection != null) return;
        UsbDeviceConnection c = usb.openDevice(device);
        if (c == null) {
            setStatus("Could not open the P2 Pro");
            return;
        }
        // Horizon OS binds the kernel uvcvideo driver to the camera, so libuvc's own claim
        // fails with BUSY. A forced claim disconnects the kernel driver; releasing right away
        // leaves the interfaces free for libuvc.
        for (int i = 0; i < device.getInterfaceCount(); i++) {
            UsbInterface intf = device.getInterface(i);
            if (c.claimInterface(intf, true)) {
                c.releaseInterface(intf);
            } else {
                Log.w(TAG, "could not detach kernel driver from interface " + intf.getId());
            }
        }
        connection = c;
        fd = c.getFileDescriptor();
        generation++;
        setStatus("P2 Pro connected");
    }

    private static void closeDevice(String reason) {
        if (connection != null) {
            connection.close();
            connection = null;
            fd = -1;
            generation++;
        }
        setStatus(reason);
    }

    private static void setStatus(String s) {
        if (!s.equals(status)) Log.i(TAG, s);
        status = s;
    }
}
