package com.dfp.runtimeapp

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

/**
 * Main activity for the Runtime Comparison app.
 *
 * This app benchmarks ExecuTorch model inference and compares performance
 * across different runtime configurations and device types.
 *
 * Usage:
 * 1. Bundle .pte model files in assets/models/
 * 2. Launch app or run instrumentation tests
 * 3. Results are saved to external files directory for retrieval
 */
class MainActivity : Activity() {

    companion object {
        private const val TAG = "RuntimeComparisonApp"
        private const val DEFAULT_MODEL = "models/kronodroid_autoencoder.pte"
    }

    private val scope = CoroutineScope(Dispatchers.Main + Job())

    private lateinit var benchmark: ModelBenchmark
    private lateinit var reporter: ResultsReporter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Simple text view for status
        val textView = TextView(this).apply {
            text = "Runtime Comparison App\n\nInitializing..."
            setPadding(32, 32, 32, 32)
            textSize = 16f
        }
        setContentView(textView)

        benchmark = ModelBenchmark(this)
        reporter = ResultsReporter(this)

        // Check if launched with specific model via intent
        val modelPath = intent.getStringExtra("model_path") ?: DEFAULT_MODEL

        // Run benchmark
        scope.launch {
            textView.text = "Loading model: $modelPath"

            val loaded = benchmark.loadModelFromAssets(modelPath)
            if (!loaded) {
                textView.text = "Failed to load model: $modelPath\n\n" +
                    "Make sure the model is bundled in assets/"
                return@launch
            }

            textView.text = "Model loaded. Running benchmark..."

            // Generate test data
            val testData = generateTestData(100, 289)

            // Run benchmark
            val results = benchmark.runInference(testData, warmupRuns = 5)

            // Save and log results
            val outputPath = reporter.saveToFile(results)
            reporter.logSummary(results)

            // Update UI
            textView.text = buildString {
                appendLine("Benchmark Complete!")
                appendLine()
                appendLine("Model: ${results.modelName}")
                appendLine("Samples: ${results.totalSamples}")
                appendLine()
                appendLine("Latency:")
                appendLine("  Avg: %.2f ms".format(results.avgLatencyMs))
                appendLine("  P50: ${results.p50LatencyMs} ms")
                appendLine("  P95: ${results.p95LatencyMs} ms")
                appendLine("  P99: ${results.p99LatencyMs} ms")
                appendLine()
                appendLine("Throughput: %.1f samples/sec".format(results.throughputSamplesPerSec))
                appendLine()
                appendLine("Results saved to:")
                appendLine(outputPath ?: "Error saving results")
            }
        }
    }

    /**
     * Generate synthetic test data for benchmarking.
     *
     * In production, this would load actual test samples.
     */
    private fun generateTestData(numSamples: Int, inputDim: Int): List<Pair<String, FloatArray>> {
        return (0 until numSamples).map { i ->
            val sampleId = "sample_%04d".format(i)
            val data = FloatArray(inputDim) { (Math.random() * 2 - 1).toFloat() }
            sampleId to data
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        benchmark.close()
    }
}
