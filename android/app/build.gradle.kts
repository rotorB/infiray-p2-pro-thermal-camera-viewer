plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.p2pro.quest"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.p2pro.quest"
        minSdk = 29
        // 32 keeps the UVC library's USB permission broadcast working (no exported-receiver
        // and mutable-PendingIntent rules from API 33/34) and is what Horizon OS expects.
        targetSdk = 32
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    packaging { jniLibs { useLegacyPackaging = true } }

    ndkVersion = "28.2.13676358"
    externalNativeBuild {
        cmake {
            path = file("../native/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation("com.herohan:UVCAndroid:1.0.13")
}
