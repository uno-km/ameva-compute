plugins {
    id("com.android.library") version "8.5.2"
    id("org.jetbrains.kotlin.android") version "1.9.24"
}

android {
    namespace = "com.ameva.amuda"
    compileSdk = 34
    ndkVersion = "26.3.11579264"

    defaultConfig {
        // 29 is a hard floor, not a preference: ac_vk_context.cpp links vkGetPhysicalDeviceProperties2/
        // Features2 (Vulkan 1.1 core) directly, and the NDK's per-API libvulkan.so stub only exports
        // them from API 29 onward (confirmed via llvm-nm against the 26.3.11579264 sysroot: 0 matches
        // at 24/26, 1 match at 29/31/33/34). All fleet devices are Android 13+ (API 33), so this costs
        // nothing in practice.
        minSdk = 29
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // Entire fleet (S25, S21, S20, A35, A53) is arm64; no other ABI is validated.
            abiFilters += "arm64-v8a"
        }
    }

    externalNativeBuild {
        cmake {
            path = file("jni/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // AGP's default (useLegacyPackaging = false) embeds .so files uncompressed inside the APK
    // zip and never extracts them to disk; ApplicationInfo.nativeLibraryDir then points at an
    // empty directory. That's fine for libraries only ever opened via System.loadLibrary(), but
    // ggml_backend_load() takes a raw filesystem path (plain dlopen) for libggml-ameva.so, which
    // needs an actual extracted file to open. Force legacy extraction so that file exists.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
}
