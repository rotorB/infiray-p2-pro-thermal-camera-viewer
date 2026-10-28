// Android library for the Unity app: USB permission bridge (Java) + libp2prouvc with the libuvc
// it loads. The AAR is copied to unity/Assets/Plugins/Android by unity/build.sh.
plugins {
    id("com.android.library")
}

// UVCAndroid's prebuilt libuvc and the two libraries it links against. Unity does not resolve
// AAR dependencies, so they are packed into this AAR instead of declared as a dependency.
val uvcAndroid: Configuration by configurations.creating
dependencies {
    uvcAndroid("com.herohan:UVCAndroid:1.0.13@aar")
}
val uvcJniDir = layout.buildDirectory.dir("uvcJni")
val extractUvcJni by tasks.registering(Copy::class) {
    from({ zipTree(uvcAndroid.singleFile) }) {
        include("jni/arm64-v8a/libuvc.so", "jni/arm64-v8a/libusb1.0.so", "jni/arm64-v8a/libjpeg-turbo212.so")
        eachFile { path = path.removePrefix("jni/") }
        includeEmptyDirs = false
    }
    into(uvcJniDir)
}

android {
    namespace = "com.p2pro.quest.unity"
    compileSdk = 35

    defaultConfig {
        minSdk = 29
        ndk { abiFilters += "arm64-v8a" }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets["main"].jniLibs.srcDir(uvcJniDir)

    ndkVersion = "28.2.13676358"
    externalNativeBuild {
        cmake {
            path = file("../native/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

tasks.named("preBuild") { dependsOn(extractUvcJni) }
