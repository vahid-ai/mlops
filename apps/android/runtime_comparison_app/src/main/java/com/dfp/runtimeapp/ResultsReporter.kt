package com.dfp.runtimeapp

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * Reports benchmark results to various destinations.
 *
 * Supports:
 * - Local file storage (JSON)
 * - HTTP POST to a results server
 * - Logcat output for debugging
 */
class ResultsReporter(private val context: Context) {

    companion object {
        private const val TAG = "ResultsReporter"
        private const val DEFAULT_FILENAME = "benchmark_results.json"
    }

    /**
     * Save benchmark results to a JSON file.
     *
     * @param results The benchmark results to save.
     * @param filename Output filename (default: benchmark_results.json).
     * @return Path to the saved file, or null if save failed.
     */
    suspend fun saveToFile(
        results: BenchmarkResults,
        filename: String = DEFAULT_FILENAME,
    ): String? = withContext(Dispatchers.IO) {
        try {
            val json = resultsToJson(results)

            // Save to app's external files directory (accessible via adb pull)
            val outputDir = context.getExternalFilesDir(null)
            val outputFile = File(outputDir, filename)
            outputFile.writeText(json.toString(2))

            Log.i(TAG, "Saved results to: ${outputFile.absolutePath}")
            outputFile.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save results", e)
            null
        }
    }

    /**
     * Post benchmark results to an HTTP endpoint.
     *
     * @param results The benchmark results to post.
     * @param endpoint URL to POST results to.
     * @return True if post was successful.
     */
    suspend fun postToServer(
        results: BenchmarkResults,
        endpoint: String,
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val json = resultsToJson(results)
            val url = URL(endpoint)

            val connection = url.openConnection() as HttpURLConnection
            connection.apply {
                requestMethod = "POST"
                setRequestProperty("Content-Type", "application/json")
                doOutput = true
                connectTimeout = 10_000
                readTimeout = 10_000
            }

            connection.outputStream.use { output ->
                output.write(json.toString().toByteArray())
            }

            val responseCode = connection.responseCode
            if (responseCode in 200..299) {
                Log.i(TAG, "Posted results to $endpoint (HTTP $responseCode)")
                true
            } else {
                Log.w(TAG, "Server returned HTTP $responseCode")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to post results to $endpoint", e)
            false
        }
    }

    /**
     * Log results summary to Logcat.
     */
    fun logSummary(results: BenchmarkResults) {
        Log.i(TAG, "=" .repeat(50))
        Log.i(TAG, "BENCHMARK RESULTS: ${results.modelName}")
        Log.i(TAG, "=" .repeat(50))
        Log.i(TAG, "Device: ${results.deviceInfo.manufacturer} ${results.deviceInfo.model}")
        Log.i(TAG, "SDK: ${results.deviceInfo.sdk}, ABI: ${results.deviceInfo.abi}")
        Log.i(TAG, "-".repeat(50))
        Log.i(TAG, "Total samples: ${results.totalSamples}")
        Log.i(TAG, "Avg latency: %.2f ms".format(results.avgLatencyMs))
        Log.i(TAG, "Min latency: ${results.minLatencyMs} ms")
        Log.i(TAG, "Max latency: ${results.maxLatencyMs} ms")
        Log.i(TAG, "P50 latency: ${results.p50LatencyMs} ms")
        Log.i(TAG, "P95 latency: ${results.p95LatencyMs} ms")
        Log.i(TAG, "P99 latency: ${results.p99LatencyMs} ms")
        Log.i(TAG, "Throughput: %.2f samples/sec".format(results.throughputSamplesPerSec))
        Log.i(TAG, "=" .repeat(50))
    }

    /**
     * Convert benchmark results to JSON.
     */
    fun resultsToJson(results: BenchmarkResults): JSONObject {
        return JSONObject().apply {
            put("modelName", results.modelName)
            put("modelPath", results.modelPath)
            put("timestamp", results.timestamp)

            put("deviceInfo", JSONObject().apply {
                put("device", results.deviceInfo.device)
                put("manufacturer", results.deviceInfo.manufacturer)
                put("model", results.deviceInfo.model)
                put("sdk", results.deviceInfo.sdk)
                put("abi", results.deviceInfo.abi)
                put("cpuCores", results.deviceInfo.cpuCores)
                put("totalMemoryMb", results.deviceInfo.totalMemoryMb)
            })

            put("metrics", JSONObject().apply {
                put("totalSamples", results.totalSamples)
                put("avgLatencyMs", results.avgLatencyMs)
                put("minLatencyMs", results.minLatencyMs)
                put("maxLatencyMs", results.maxLatencyMs)
                put("p50LatencyMs", results.p50LatencyMs)
                put("p95LatencyMs", results.p95LatencyMs)
                put("p99LatencyMs", results.p99LatencyMs)
                put("throughputSamplesPerSec", results.throughputSamplesPerSec)
            })

            put("results", JSONArray().apply {
                results.results.forEach { result ->
                    put(JSONObject().apply {
                        put("sampleId", result.sampleId)
                        put("latencyMs", result.latencyMs)
                        put("outputs", JSONArray(result.outputs))
                    })
                }
            })
        }
    }

    /**
     * Parse benchmark results from JSON file.
     */
    fun parseFromJson(json: JSONObject): BenchmarkResults {
        val deviceInfoJson = json.getJSONObject("deviceInfo")
        val metricsJson = json.getJSONObject("metrics")
        val resultsJson = json.getJSONArray("results")

        val deviceInfo = DeviceInfo(
            device = deviceInfoJson.getString("device"),
            manufacturer = deviceInfoJson.getString("manufacturer"),
            model = deviceInfoJson.getString("model"),
            sdk = deviceInfoJson.getInt("sdk"),
            abi = deviceInfoJson.getString("abi"),
            cpuCores = deviceInfoJson.getInt("cpuCores"),
            totalMemoryMb = deviceInfoJson.getLong("totalMemoryMb"),
        )

        val results = (0 until resultsJson.length()).map { i ->
            val resultJson = resultsJson.getJSONObject(i)
            val outputsJson = resultJson.getJSONArray("outputs")
            InferenceResult(
                sampleId = resultJson.getString("sampleId"),
                latencyMs = resultJson.getLong("latencyMs"),
                outputs = (0 until outputsJson.length()).map { j ->
                    outputsJson.getDouble(j).toFloat()
                },
            )
        }

        return BenchmarkResults(
            modelName = json.getString("modelName"),
            modelPath = json.getString("modelPath"),
            deviceInfo = deviceInfo,
            results = results,
            avgLatencyMs = metricsJson.getDouble("avgLatencyMs"),
            minLatencyMs = metricsJson.getLong("minLatencyMs"),
            maxLatencyMs = metricsJson.getLong("maxLatencyMs"),
            p50LatencyMs = metricsJson.getLong("p50LatencyMs"),
            p95LatencyMs = metricsJson.getLong("p95LatencyMs"),
            p99LatencyMs = metricsJson.getLong("p99LatencyMs"),
            totalSamples = metricsJson.getInt("totalSamples"),
            throughputSamplesPerSec = metricsJson.getDouble("throughputSamplesPerSec"),
            timestamp = json.getLong("timestamp"),
        )
    }
}
