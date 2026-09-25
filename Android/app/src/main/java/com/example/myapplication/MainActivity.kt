package com.example.myapplication

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var viewFinder: PreviewView
    private lateinit var tts: TextToSpeech
    private var lastSpokenText = ""

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) startCamera()
        else Toast.makeText(this, "Camera permission required.", Toast.LENGTH_SHORT).show()
    }

    private fun copyAssetToStorage(filename: String): String {
        val file = File(filesDir, filename)
        if (!file.exists()) {
            assets.open(filename).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return file.absolutePath
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        viewFinder = findViewById(R.id.viewFinder)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize the Android Speech Engine
        tts = TextToSpeech(this, this)

        val yoloPath = copyAssetToStorage("yolov8n_hazards.onnx")
        val deeplabPath = copyAssetToStorage("deeplabv3_mobilenet_safepath.onnx")
        copyAssetToStorage("deeplabv3_mobilenet_safepath.onnx.data")
        val depthPath = copyAssetToStorage("depth_anything_v2_small.onnx")

        initEngine(yoloPath, deeplabPath, depthPath)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // Configure the Voice
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.US
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        val buffer = imageProxy.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)

                        // Push pixels to C++ and immediately get the steering decision back
                        val decision = processFrame(imageProxy.width, imageProxy.height, bytes)

                        // Only speak if the hazard state has changed to prevent audio spam
                        if (decision.isNotBlank() && decision != lastSpokenText && decision != "BOOTING SYSTEM") {
                            tts.speak(decision, TextToSpeech.QUEUE_FLUSH, null, null)
                            lastSpokenText = decision
                        }

                        imageProxy.close()
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalyzer)
            } catch(exc: Exception) {
                Log.e("SafePath", "Camera bind failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    external fun initEngine(yoloPath: String, deeplabPath: String, depthPath: String): Boolean

    // Notice that the bridge now returns a String!
    external fun processFrame(width: Int, height: Int, data: ByteArray): String

    companion object {
        init {
            System.loadLibrary("onnxruntime")
            System.loadLibrary("myapplication")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.stop()
        tts.shutdown()
        cameraExecutor.shutdown()
    }
}