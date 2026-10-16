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

val sherpaDir = layout.buildDirectory.dir("sherpa-onnx")
val sherpaAar = file("libs/sherpa-onnx-1.13.8.aar")
// A separate task: zipTree() is resolved while Gradle snapshots the Sync task's inputs, i.e. before
// any doFirst of that task, so the explicit error has to come from a prerequisite task.
val checkSherpaAar = tasks.register("checkSherpaAar") {
    doLast {
        if (!sherpaAar.isFile) {
            throw GradleException(
                "AMU_ERR_MISSING_SHERPA_AAR: ${sherpaAar} not found. It is not committed; run " +
                    "'bash android/scripts/fetch_sherpa_onnx.sh' (downloads upstream v1.13.8 and verifies sha256)."
            )
        }
    }
}
val extractSherpaOnnx = tasks.register<Sync>("extractSherpaOnnx") {
    dependsOn(checkSherpaAar)
    from(zipTree(sherpaAar)) {
        include("classes.jar", "jni/arm64-v8a/*.so")
    }
    into(sherpaDir)
}
android.sourceSets["main"].jniLibs.srcDir(sherpaDir.map { it.dir("jni") })
tasks.named("preBuild") { dependsOn(extractSherpaOnnx) }

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    // k2-fsa/sherpa-onnx v1.13.8 (CPU/ONNX Runtime, not ggml-ameva -- no AMEVA GPU acceleration on
    // this path, see AmevaSherpaStt.kt). No Maven Central/JitPack coordinates exist for this
    // project; upstream only publishes prebuilt .aar files as GitHub Release assets. The .aar is
    // not committed (~48 MiB, gitignored): android/scripts/fetch_sherpa_onnx.sh downloads the pinned
    // release into libs/ and verifies its sha256 (CI runs it before the build; extractSherpaOnnx
    // fails with AMU_ERR_MISSING_SHERPA_AAR if it was not run). Covers SenseVoice, Moonshine,
    // streaming Zipformer, and speaker diarization in one dependency --
    // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.8
    //
    // AGP refuses to bundle a local .aar into an .aar ("Direct local .aar file dependencies are not
    // supported when building an AAR"), so the downloaded .aar is unpacked by extractSherpaOnnx above:
    // its classes.jar is bundled as a local jar and its arm64-v8a .so files are added as jniLibs.
    // The AAR is therefore self-contained; consumers must not also add the upstream sherpa-onnx .aar.
    implementation(files(sherpaDir.map { it.file("classes.jar") }).builtBy(extractSherpaOnnx))
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
}
