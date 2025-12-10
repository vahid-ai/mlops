package com.dfp.runtime

class ExecuRunner {
    init {
        System.loadLibrary("execu_bridge")
    }

    external fun run(modelPath: String): Int
}
