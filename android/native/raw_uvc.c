/*
 * Raw UVC capture for the P2 Pro through the libuvc that ships inside UVCAndroid.
 *
 * UVCAndroid's own frame callback converts every YUYV frame to RGBX and back, which destroys
 * the 16-bit temperature words in the bottom half. Here frames go from libuvc straight into a
 * buffer, byte for byte. libuvc.so is resolved with dlopen so this library does not link
 * against (or re-package) UVCAndroid's prebuilt .so files.
 *
 * One camera at a time: libuvc calls on_frame from its own thread. Consumers either block in
 * nativeWaitFrame (Kotlin viewer) or poll p2pro_take_celsius once per rendered frame (Unity,
 * through P/Invoke). The p2pro_* functions are the plain C API; the JNI functions wrap them.
 */
#include <android/log.h>
#include <dlfcn.h>
#include <errno.h>
#include <jni.h>
#include <linux/usbdevice_fs.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define TAG "P2ProQuest"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/* Only the libuvc API this file calls; signatures match UVCAndroid 1.0.13's libuvc.h. */
typedef struct uvc_context uvc_context_t;
typedef struct uvc_device_handle uvc_device_handle_t;
/* Leading fields of struct uvc_frame; the rest is never read. */
typedef struct {
    void *data;
    size_t data_bytes;
} uvc_frame_head_t;
typedef void(uvc_frame_callback_t)(void *frame, void *user_ptr);

#define UVC_FRAME_FORMAT_YUYV 3
#define ERR_NO_LIBUVC (-100)
#define ERR_NO_MEMORY (-101)
#define ERR_ALREADY_OPEN (-102)

static struct {
    int (*init2)(uvc_context_t **, void *);
    void (*exit)(uvc_context_t *);
    int (*wrap)(int, uvc_context_t *, uvc_device_handle_t **);
    void (*close)(uvc_device_handle_t *);
    int (*get_stream_ctrl_format_size)(uvc_device_handle_t *, void *, int, int, int, int);
    int (*start_streaming)(uvc_device_handle_t *, void *, uvc_frame_callback_t *, void *, uint8_t);
    void (*stop_streaming)(uvc_device_handle_t *);
    const char *(*strerror)(int);
} uvc;

static struct {
    pthread_mutex_t lock;
    pthread_cond_t ready;
    uvc_context_t *ctx;
    uvc_device_handle_t *devh;
    int fd;
    int running;
    uint8_t *frame;         /* latest complete frame, guarded by lock */
    size_t frame_bytes;
    uint64_t sequence;      /* bumped for every complete frame */
    int64_t frame_time_ns;  /* CLOCK_MONOTONIC when the latest frame was complete */
    uint64_t delivered;     /* sequence last handed to Java */
    uint64_t short_frames;
    /* uvc_stream_ctrl_t is ~48 bytes; leave room in case the library's copy is larger. */
    uint8_t ctrl[256] __attribute__((aligned(8)));
} cam = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .ready = PTHREAD_COND_INITIALIZER,
    .fd = -1,
};

static int load_libuvc(void) {
    if (uvc.start_streaming) return 0;
    void *lib = dlopen("libuvc.so", RTLD_NOW);
    if (!lib) {
        LOGE("dlopen libuvc.so: %s", dlerror());
        return ERR_NO_LIBUVC;
    }
#define SYM(field, name) if (!(*(void **) &uvc.field = dlsym(lib, name))) { LOGE("missing %s", name); return ERR_NO_LIBUVC; }
    SYM(init2, "uvc_init2")
    SYM(exit, "uvc_exit")
    SYM(wrap, "uvc_wrap")
    SYM(close, "uvc_close")
    SYM(get_stream_ctrl_format_size, "uvc_get_stream_ctrl_format_size")
    SYM(stop_streaming, "uvc_stop_streaming")
    SYM(strerror, "uvc_strerror")
    SYM(start_streaming, "uvc_start_streaming")  /* last: marks the table complete */
#undef SYM
    return 0;
}

static int64_t monotonic_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t) t.tv_sec * 1000000000LL + t.tv_nsec;
}

static void on_frame(void *frame, void *user_ptr) {
    (void) user_ptr;
    const uvc_frame_head_t *f = frame;
    int64_t now = monotonic_ns();
    pthread_mutex_lock(&cam.lock);
    if (cam.frame && f->data_bytes >= cam.frame_bytes) {
        memcpy(cam.frame, f->data, cam.frame_bytes);
        cam.frame_time_ns = now;
        cam.sequence++;
        pthread_cond_signal(&cam.ready);
    } else if (cam.short_frames++ % 100 == 0) {
        LOGI("short frame: %zu of %zu bytes", f->data_bytes, cam.frame_bytes);
    }
    pthread_mutex_unlock(&cam.lock);
}

/* Undo whatever nativeOpen set up; streaming must already be stopped. */
static void release(void) {
    if (cam.devh) uvc.close(cam.devh);
    if (cam.ctx) uvc.exit(cam.ctx);
    if (cam.fd >= 0) close(cam.fd);
    cam.devh = NULL;
    cam.ctx = NULL;
    cam.fd = -1;
    pthread_mutex_lock(&cam.lock);
    free(cam.frame);
    cam.frame = NULL;
    pthread_mutex_unlock(&cam.lock);
}

#define P2PRO_API __attribute__((visibility("default")))

/* Starts YUYV streaming on the device behind `fd` (dup'ed). Returns 0 or a libuvc error code. */
P2PRO_API int p2pro_open(int fd, int width, int height, int fps) {
    int rc = load_libuvc();
    if (rc) return rc;
    if (cam.devh) return ERR_ALREADY_OPEN;

    if ((rc = uvc.init2(&cam.ctx, NULL)) < 0) {
        LOGE("uvc_init2: %s", uvc.strerror(rc));
        cam.ctx = NULL;
        return rc;
    }
    /* libuvc keeps the descriptor; the Java side may close its UsbDeviceConnection copy. */
    cam.fd = dup(fd);
    if ((rc = uvc.wrap(cam.fd, cam.ctx, &cam.devh)) < 0) {
        LOGE("uvc_wrap: %s", uvc.strerror(rc));
        cam.devh = NULL;
        release();
        return rc;
    }
    memset(cam.ctrl, 0, sizeof cam.ctrl);
    if ((rc = uvc.get_stream_ctrl_format_size(cam.devh, cam.ctrl, UVC_FRAME_FORMAT_YUYV, width, height, fps)) < 0) {
        LOGE("no YUYV %dx%d@%d: %s", width, height, fps, uvc.strerror(rc));
        release();
        return rc;
    }

    pthread_mutex_lock(&cam.lock);
    cam.frame_bytes = (size_t) width * height * 2;
    cam.frame = malloc(cam.frame_bytes);
    cam.sequence = cam.delivered = cam.short_frames = 0;
    cam.running = cam.frame != NULL;
    pthread_mutex_unlock(&cam.lock);
    if (!cam.frame) {
        release();
        return ERR_NO_MEMORY;
    }

    if ((rc = uvc.start_streaming(cam.devh, cam.ctrl, on_frame, NULL, 0)) < 0) {
        LOGE("uvc_start_streaming: %s", uvc.strerror(rc));
        pthread_mutex_lock(&cam.lock);
        cam.running = 0;
        pthread_mutex_unlock(&cam.lock);
        release();
        return rc;
    }
    LOGI("raw YUYV %dx%d@%d streaming", width, height, fps);
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_p2pro_quest_RawUvc_nativeOpen(JNIEnv *env, jobject thiz, jint fd, jint width, jint height, jint fps) {
    (void) env; (void) thiz;
    return p2pro_open(fd, width, height, fps);
}

/*
 * InfiRay vendor command channel (vendor control requests on endpoint 0, sent straight to the
 * USB descriptor next to libuvc's streaming). Transport from LeoDJ/P2Pro-Viewer; the long-form
 * framing and the auto-shutter parameter from stanelie/p2-viewer (extracted from the official
 * app's libircmd.so). Only the auto-shutter switch is used: it is volatile (the camera forgets
 * it when it loses power) and never touches flash. Never add spi_transfer (0x8201) or
 * sys_reset_to_rom (0x0805) here.
 */
#define VC_OUT 0x41
#define VC_IN 0xC1
#define VC_WRITE 0x45
#define VC_READ 0x44
#define VC_VALUE 0x78
#define VC_STATUS 0x0200        /* bits 0-1 busy, 2-7 error */
#define VC_DATA 0x1d08
#define VC_CMD_LONG 0x9d00
#define VC_DATA_LONG 0x1d10
#define CMD_GET_AUTO_SHUTTER 0x8214
#define CMD_SET_AUTO_SHUTTER 0xc214
#define AUTO_SHUTTER_SWITCH 0
#define ERR_NOT_OPEN (-103)
#define ERR_VC_STATUS (-104)
#define ERR_VC_TIMEOUT (-105)

static int vc_transfer(uint8_t type, uint8_t request, uint16_t index, void *data, uint16_t length) {
    struct usbdevfs_ctrltransfer t = {
        .bRequestType = type, .bRequest = request, .wValue = VC_VALUE, .wIndex = index,
        .wLength = length, .timeout = 1000, .data = data,
    };
    return ioctl(cam.fd, USBDEVFS_CONTROL, &t) < 0 ? -errno : 0;
}

static int vc_wait_ready(void) {
    for (int waited = 0; waited < 1000; waited += 2) {
        uint8_t status = 0xFF;
        int rc = vc_transfer(VC_IN, VC_READ, VC_STATUS, &status, 1);
        if (rc) return rc;
        if ((status & 0x03) == 0) return 0;
        if (status & 0xFC) {
            LOGE("vendor command status error 0x%02x", status);
            return ERR_VC_STATUS;
        }
        usleep(2000);
    }
    return ERR_VC_TIMEOUT;
}

/* Long form: cmd LE, p1 BE16, p2 BE32 to 0x9d00; then 8 bytes to 0x1d08 (read length BE32 in
 * the last four for a read, zeros for a write); poll ready; a read then fetches 0x1d10. */
static int vc_long(uint16_t cmd, uint16_t p1, uint32_t p2, uint8_t *out, uint16_t out_len) {
    uint8_t head[8] = {
        cmd & 0xFF, cmd >> 8, p1 >> 8, p1 & 0xFF, p2 >> 24, (p2 >> 16) & 0xFF, (p2 >> 8) & 0xFF, p2 & 0xFF,
    };
    uint8_t tail[8] = {0, 0, 0, 0, 0, 0, out_len >> 8, out_len & 0xFF};
    int rc;
    if ((rc = vc_transfer(VC_OUT, VC_WRITE, VC_CMD_LONG, head, sizeof head))) return rc;
    if ((rc = vc_transfer(VC_OUT, VC_WRITE, VC_DATA, tail, sizeof tail))) return rc;
    if ((rc = vc_wait_ready())) return rc;
    return out ? vc_transfer(VC_IN, VC_READ, VC_DATA_LONG, out, out_len) : 0;
}

/*
 * Turns the camera's automatic shutter (its periodic flat-field calibration) on or off and reads
 * the switch back. Returns the switch as read back (0 or 1), or a negative error. The camera only
 * answers a few seconds after streaming starts, and forgets the setting when it loses power.
 * Blocks up to a few seconds: call it off the render thread.
 */
P2PRO_API int p2pro_set_auto_shutter(int on) {
    if (cam.fd < 0 || !cam.running) return ERR_NOT_OPEN;
    int rc = vc_long(CMD_SET_AUTO_SHUTTER, AUTO_SHUTTER_SWITCH, on ? 1 : 0, NULL, 0);
    if (rc) {
        LOGE("auto-shutter := %d: error %d", on, rc);
        return rc;
    }
    uint8_t back[2] = {0xFF, 0xFF};
    if ((rc = vc_long(CMD_GET_AUTO_SHUTTER, AUTO_SHUTTER_SWITCH, 0, back, sizeof back))) {
        LOGE("auto-shutter read back: error %d", rc);
        return rc;
    }
    int value = back[0] << 8 | back[1];
    LOGI("auto-shutter := %d, reads back %d", on, value);
    return value;
}

/*
 * Copies the next unseen frame's temperature half into `dst` as °C, `width` x `height` values
 * (256 x 192 for the P2 Pro), rows flipped so row 0 is the bottom of the image (texture order).
 * `stats` receives min, max and the centre pixel in °C, `time_ns` the CLOCK_MONOTONIC time the
 * frame finished arriving over USB (the same time base as OpenXR's XrTime on Quest; the sensor
 * exposure is earlier by the camera's latency). Returns 1 for a new frame, 0 when there is none
 * yet, -1 when the camera is not streaming or the size does not match.
 */
P2PRO_API int p2pro_take_celsius(float *dst, int width, int height, float *stats, int64_t *time_ns) {
    int result;
    pthread_mutex_lock(&cam.lock);
    if (!cam.running) {
        result = -1;
    } else if ((size_t) width * height * 4 != cam.frame_bytes) {
        LOGE("temperature size %dx%d does not match the %zu byte frame", width, height, cam.frame_bytes);
        result = -1;
    } else if (cam.sequence == cam.delivered) {
        result = 0;
    } else {
        /* Bottom half of the YUYV frame: uint16 little endian, Kelvin x 64. */
        const uint8_t *raw = cam.frame + cam.frame_bytes / 2;
        float lo = 1e9f, hi = -1e9f;
        for (int y = 0; y < height; y++) {
            const uint8_t *src = raw + (size_t) y * width * 2;
            float *row = dst + (size_t) (height - 1 - y) * width;
            for (int x = 0; x < width; x++) {
                float c = (float) (src[2 * x] | src[2 * x + 1] << 8) / 64.0f - 273.15f;
                row[x] = c;
                if (c < lo) lo = c;
                if (c > hi) hi = c;
            }
        }
        stats[0] = lo;
        stats[1] = hi;
        stats[2] = dst[(size_t) (height / 2) * width + width / 2];
        *time_ns = cam.frame_time_ns;
        cam.delivered = cam.sequence;
        result = 1;
    }
    pthread_mutex_unlock(&cam.lock);
    return result;
}

/*
 * Copies the next unseen frame into `buffer` (a direct ByteBuffer). Returns its size,
 * 0 on timeout, or -1 once the camera is closed.
 */
JNIEXPORT jint JNICALL
Java_com_p2pro_quest_RawUvc_nativeWaitFrame(JNIEnv *env, jobject thiz, jobject buffer, jint timeout_ms) {
    (void) thiz;
    uint8_t *dst = (*env)->GetDirectBufferAddress(env, buffer);
    jlong capacity = (*env)->GetDirectBufferCapacity(env, buffer);

    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += timeout_ms / 1000;
    deadline.tv_nsec += (long) (timeout_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }

    jint result;
    pthread_mutex_lock(&cam.lock);
    while (cam.running && cam.sequence == cam.delivered) {
        if (pthread_cond_timedwait(&cam.ready, &cam.lock, &deadline) == ETIMEDOUT) break;
    }
    if (!cam.running) {
        result = -1;
    } else if (cam.sequence == cam.delivered) {
        result = 0;
    } else if (!dst || capacity < (jlong) cam.frame_bytes) {
        LOGE("frame buffer too small: %lld < %zu", (long long) capacity, cam.frame_bytes);
        result = -1;
    } else {
        memcpy(dst, cam.frame, cam.frame_bytes);
        cam.delivered = cam.sequence;
        result = (jint) cam.frame_bytes;
    }
    pthread_mutex_unlock(&cam.lock);
    return result;
}

/* CLOCK_MONOTONIC now, to compare the frame time base with other clocks. */
P2PRO_API int64_t p2pro_now_ns(void) {
    return monotonic_ns();
}

/* Stops streaming and wakes any waiting reader. Safe to call when nothing is open. */
P2PRO_API void p2pro_close(void) {
    if (!cam.devh) return;
    pthread_mutex_lock(&cam.lock);
    cam.running = 0;
    pthread_cond_broadcast(&cam.ready);
    pthread_mutex_unlock(&cam.lock);
    /* Not under the lock: stopping waits for the callback thread, which takes the lock. */
    uvc.stop_streaming(cam.devh);
    release();
    LOGI("raw stream closed");
}

JNIEXPORT void JNICALL
Java_com_p2pro_quest_RawUvc_nativeClose(JNIEnv *env, jobject thiz) {
    (void) env; (void) thiz;
    p2pro_close();
}
