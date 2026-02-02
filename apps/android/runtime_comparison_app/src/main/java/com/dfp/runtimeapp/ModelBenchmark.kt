package com.dfp.runtimeapp

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream

/**
 * Result of a single inference run.
 *
 * @property sampleId Unique identifier for the test sample.
 * @property outputs Model output values.
 * @property latencyMs Inference latency in milliseconds.
 */
data class InferenceResult(
    val sampleId: String,
    val outputs: List<Float>,
    val latencyMs: Long,
)

/**
 * Information about the device running the benchmark.
 */
data class DeviceInfo(
    val device: String,
    val manufacturer: String,
    val model: String,
    val sdk: Int,
    val abi: String,
    val cpuCores: Int,
    val totalMemoryMb: Long,
)

/**
 * Complete benchmark results including all inference runs.
 *
 * @property modelName Name of the model being benchmarked.
 * @property modelPath Path to the model file.
 * @property deviceInfo Information about the test device.
 * @property results Individual inference results.
 * @property avgLatencyMs Average inference latency.
 * @property minLatencyMs Minimum inference latency.
 * @property maxLatencyMs Maximum inference latency.
 * @property p50LatencyMs 50th percentile latency.
 * @property p95LatencyMs 95th percentile latency.
 * @property p99LatencyMs 99th percentile latency.
 * @property totalSamples Total number of samples processed.
 * @property throughputSamplesPerSec Inference throughput.
 */
data class BenchmarkResults(
    val modelName: String,
    val modelPath: String,
    val deviceInfo: DeviceInfo,
    val results: List<InferenceResult>,
    val avgLatencyMs: Double,
    val minLatencyMs: Long,
    val maxLatencyMs: Long,
    val p50LatencyMs: Long,
    val p95LatencyMs: Long,
    val p99LatencyMs: Long,
    val totalSamples: Int,
    val throughputSamplesPerSec: Double,
    val timestamp: Long = System.currentTimeMillis(),
)

/**
 * Benchmarks ExecuTorch model inference on Android.
 *
 * Provides utilities to:
 * - Load ExecuTorch models from assets or files
 * - Run inference on test data
 * - Collect detailed performance metrics
 * - Export results for analysis
 */
class ModelBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "ModelBenchmark"
    }

    private var module: Module? = null
    private var modelName: String = "unknown"
    private var modelPath: String = ""

    /**
     * Load an ExecuTorch model from assets.
     *
     * @param assetPath Path to the model within the assets folder.
     * @return True if model loaded successfully.
     */
    suspend fun loadModelFromAssets(assetPath: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val modelFile = assetFilePath(context, assetPath)
            module = Module.load(modelFile)
            modelName = File(assetPath).nameWithoutExtension
            modelPath = assetPath
            Log.i(TAG, "Loaded model from assets: $assetPath")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model from assets: $assetPath", e)
            false
        }
    }

    /**
     * Load an ExecuTorch model from a file path.
     *
     * @param filePath Absolute path to the .pte file.
     * @return True if model loaded successfully.
     */
    suspend fun loadModelFromFile(filePath: String): Boolean = withContext(Dispatchers.IO) {
        try {
            module = Module.load(filePath)
            modelName = File(filePath).nameWithoutExtension
            modelPath = filePath
            Log.i(TAG, "Loaded model from file: $filePath")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model from file: $filePath", e)
            false
        }
    }

    /**
     * Run inference on a batch of test samples.
     *
     * @param inputs List of (sampleId, inputData) pairs.
     * @param warmupRuns Number of warmup runs before timing (default: 3).
     * @return BenchmarkResults with detailed metrics.
     */
    suspend fun runInference(
        inputs: List<Pair<String, FloatArray>>,
        warmupRuns: Int = 3,
    ): BenchmarkResults = withContext(Dispatchers.Default) {
        val mod = module ?: throw IllegalStateException("Model not loaded. Call loadModel first.")

        // Warmup runs (not timed)
        if (inputs.isNotEmpty() && warmupRuns > 0) {
            Log.d(TAG, "Running $warmupRuns warmup iterations...")
            val warmupInput = inputs.first().second
            repeat(warmupRuns) {
                runSingleInference(mod, warmupInput)
            }
        }

        // Timed runs
        Log.d(TAG, "Running inference on ${inputs.size} samples...")
        val results = inputs.map { (sampleId, inputData) ->
            val startTime = System.nanoTime()
            val outputs = runSingleInference(mod, inputData)
            val latencyMs = (System.nanoTime() - startTime) / 1_000_000

            InferenceResult(
                sampleId = sampleId,
                outputs = outputs,
                latencyMs = latencyMs,
            )
        }

        // Calculate statistics
        val latencies = results.map { it.latencyMs }.sorted()
        val totalTimeMs = latencies.sum()

        BenchmarkResults(
            modelName = modelName,
            modelPath = modelPath,
            deviceInfo = getDeviceInfo(),
            results = results,
            avgLatencyMs = latencies.average(),
            minLatencyMs = latencies.minOrNull() ?: 0,
            maxLatencyMs = latencies.maxOrNull() ?: 0,
            p50LatencyMs = percentile(latencies, 50),
            p95LatencyMs = percentile(latencies, 95),
            p99LatencyMs = percentile(latencies, 99),
            totalSamples = results.size,
            throughputSamplesPerSec = if (totalTimeMs > 0) {
                results.size * 1000.0 / totalTimeMs
            } else {
                0.0
            },
        )
    }

    /**
     * Run inference on a single input and return outputs.
     */
    private fun runSingleInference(mod: Module, inputData: FloatArray): List<Float> {
        val inputTensor = Tensor.fromBlob(
            inputData,
            longArrayOf(1, inputData.size.toLong())
        )

        val output = mod.forward(EValue.from(inputTensor))
        val outputTensor = output[0].toTensor()

        return outputTensor.dataAsFloatArray.toList()
    }

    /**
     * Get information about the current device.
     */
    private fun getDeviceInfo(): DeviceInfo {
        val runtime = Runtime.getRuntime()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        return DeviceInfo(
            device = android.os.Build.DEVICE,
            manufacturer = android.os.Build.MANUFACTURER,
            model = android.os.Build.MODEL,
            sdk = android.os.Build.VERSION.SDK_INT,
            abi = android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
            cpuCores = runtime.availableProcessors(),
            totalMemoryMb = memInfo.totalMem / (1024 * 1024),
        )
    }

    /**
     * Calculate percentile from sorted list.
     */
    private fun percentile(sortedValues: List<Long>, percentile: Int): Long {
        if (sortedValues.isEmpty()) return 0
        val index = (percentile / 100.0 * (sortedValues.size - 1)).toInt()
        return sortedValues[index.coerceIn(0, sortedValues.lastIndex)]
    }

    /**
     * Copy asset to internal storage and return path.
     */
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        // Create parent directories if needed
        file.parentFile?.mkdirs()

        if (!file.exists()) {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }

        return file.absolutePath
    }

    /**
     * Release model resources.
     */
    fun close() {
        module = null
    }
}
