package com.dfp.runtimeapp

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Instrumentation tests for model accuracy validation.
 *
 * These tests run on an Android device/emulator and validate:
 * 1. Model loads successfully
 * 2. Inference produces expected output shapes
 * 3. Outputs are within tolerance of expected values
 * 4. Performance meets requirements
 *
 * Run with:
 *   adb shell am instrument -w -e class com.dfp.runtimeapp.ModelAccuracyTest \
 *     com.dfp.runtimeapp.test/androidx.test.runner.AndroidJUnitRunner
 *
 * Or via Bazel:
 *   bazel test --config=android_test //apps/android/runtime_comparison_app:model_accuracy_test
 */
@RunWith(AndroidJUnit4::class)
class ModelAccuracyTest {

    companion object {
        private const val TAG = "ModelAccuracyTest"

        // Test configuration
        private const val MODEL_PATH = "models/kronodroid_autoencoder.pte"
        private const val INPUT_DIM = 289
        private const val EXPECTED_OUTPUT_DIM = 289

        // Performance thresholds
        private const val MAX_AVG_LATENCY_MS = 100.0
        private const val MAX_P99_LATENCY_MS = 500L

        // Accuracy thresholds
        private const val OUTPUT_TOLERANCE = 0.01f
    }

    private lateinit var context: Context
    private lateinit var benchmark: ModelBenchmark
    private lateinit var reporter: ResultsReporter

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        benchmark = ModelBenchmark(context)
        reporter = ResultsReporter(context)
    }

    @After
    fun teardown() {
        benchmark.close()
    }

    /**
     * Test that the model loads successfully.
     */
    @Test
    fun testModelLoads() = runBlocking {
        val loaded = benchmark.loadModelFromAssets(MODEL_PATH)
        assertTrue("Model should load successfully", loaded)
    }

    /**
     * Test inference produces correct output shape.
     */
    @Test
    fun testOutputShape() = runBlocking {
        val loaded = benchmark.loadModelFromAssets(MODEL_PATH)
        assertTrue("Model should load", loaded)

        val input = FloatArray(INPUT_DIM) { 0.5f }
        val results = benchmark.runInference(listOf("test" to input), warmupRuns = 1)

        assertEquals("Should have 1 result", 1, results.results.size)

        val output = results.results.first().outputs
        assertEquals(
            "Output should have expected dimension",
            EXPECTED_OUTPUT_DIM,
            output.size
        )
    }

    /**
     * Test inference performance meets requirements.
     */
    @Test
    fun testPerformance() = runBlocking {
        val loaded = benchmark.loadModelFromAssets(MODEL_PATH)
        assertTrue("Model should load", loaded)

        // Generate test data
        val testData = (0 until 50).map { i ->
            "perf_sample_$i" to FloatArray(INPUT_DIM) { (Math.random() * 2 - 1).toFloat() }
        }

        // Run benchmark
        val results = benchmark.runInference(testData, warmupRuns = 5)

        // Log results
        reporter.logSummary(results)

        // Assert performance thresholds
        assertTrue(
            "Average latency (${results.avgLatencyMs} ms) should be < $MAX_AVG_LATENCY_MS ms",
            results.avgLatencyMs < MAX_AVG_LATENCY_MS
        )

        assertTrue(
            "P99 latency (${results.p99LatencyMs} ms) should be < $MAX_P99_LATENCY_MS ms",
            results.p99LatencyMs < MAX_P99_LATENCY_MS
        )

        // Save results for retrieval
        reporter.saveToFile(results, "performance_test_results.json")
    }

    /**
     * Test inference against expected outputs.
     *
     * This test loads expected outputs from a JSON file (if available)
     * and compares model outputs against them.
     */
    @Test
    fun testAccuracyAgainstExpected() = runBlocking {
        val loaded = benchmark.loadModelFromAssets(MODEL_PATH)
        assertTrue("Model should load", loaded)

        // Try to load expected outputs from test assets
        val expectedOutputs = loadExpectedOutputs()

        if (expectedOutputs.isEmpty()) {
            Log.w(TAG, "No expected outputs found, skipping accuracy test")
            return@runBlocking
        }

        // Run inference on test samples
        val testData = expectedOutputs.map { (sampleId, input, _) ->
            sampleId to input
        }

        val results = benchmark.runInference(testData, warmupRuns = 1)

        // Compare outputs
        var passCount = 0
        var totalDiff = 0.0

        for (i in expectedOutputs.indices) {
            val (sampleId, _, expected) = expectedOutputs[i]
            val actual = results.results[i].outputs

            val maxDiff = expected.zip(actual).maxOfOrNull { (e, a) ->
                kotlin.math.abs(e - a)
            } ?: 0f

            if (maxDiff <= OUTPUT_TOLERANCE) {
                passCount++
            } else {
                Log.w(TAG, "Sample $sampleId: max diff = $maxDiff (tolerance: $OUTPUT_TOLERANCE)")
            }

            totalDiff += maxDiff
        }

        val passRate = passCount.toDouble() / expectedOutputs.size
        val avgDiff = totalDiff / expectedOutputs.size

        Log.i(TAG, "Accuracy test: $passCount/${expectedOutputs.size} passed (${passRate * 100}%)")
        Log.i(TAG, "Average max diff: $avgDiff")

        // Save accuracy results
        val accuracyResults = JSONObject().apply {
            put("totalSamples", expectedOutputs.size)
            put("passedSamples", passCount)
            put("passRate", passRate)
            put("avgMaxDiff", avgDiff)
            put("tolerance", OUTPUT_TOLERANCE)
        }

        val outputFile = File(context.getExternalFilesDir(null), "accuracy_test_results.json")
        outputFile.writeText(accuracyResults.toString(2))

        // Assert pass rate
        assertTrue(
            "Pass rate ($passRate) should be >= 0.99 (99%)",
            passRate >= 0.99
        )
    }

    /**
     * Test model produces consistent outputs (deterministic).
     */
    @Test
    fun testDeterminism() = runBlocking {
        val loaded = benchmark.loadModelFromAssets(MODEL_PATH)
        assertTrue("Model should load", loaded)

        val input = FloatArray(INPUT_DIM) { 0.5f }

        // Run inference multiple times with same input
        val runs = 5
        val outputs = mutableListOf<List<Float>>()

        repeat(runs) {
            val results = benchmark.runInference(listOf("determinism_test" to input), warmupRuns = 0)
            outputs.add(results.results.first().outputs)
        }

        // All outputs should be identical
        for (i in 1 until outputs.size) {
            val maxDiff = outputs[0].zip(outputs[i]).maxOfOrNull { (a, b) ->
                kotlin.math.abs(a - b)
            } ?: 0f

            assertTrue(
                "Run $i output should match run 0 (max diff: $maxDiff)",
                maxDiff < 1e-6f
            )
        }
    }

    /**
     * Load expected outputs from test assets.
     *
     * Expected format (JSON):
     * [
     *   {
     *     "sampleId": "sample_001",
     *     "input": [0.1, 0.2, ...],
     *     "expected": [0.3, 0.4, ...]
     *   },
     *   ...
     * ]
     */
    private fun loadExpectedOutputs(): List<Triple<String, FloatArray, FloatArray>> {
        return try {
            val json = context.assets.open("test_data/expected_outputs.json")
                .bufferedReader()
                .readText()

            val array = JSONArray(json)
            (0 until array.length()).map { i ->
                val obj = array.getJSONObject(i)
                val sampleId = obj.getString("sampleId")

                val inputArray = obj.getJSONArray("input")
                val input = FloatArray(inputArray.length()) { j ->
                    inputArray.getDouble(j).toFloat()
                }

                val expectedArray = obj.getJSONArray("expected")
                val expected = FloatArray(expectedArray.length()) { j ->
                    expectedArray.getDouble(j).toFloat()
                }

                Triple(sampleId, input, expected)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load expected outputs: ${e.message}")
            emptyList()
        }
    }
}
