# VHS recording tape for PAW-Kit (paw-kit)
# Record via: vhs demo.tape

Output assets/demo.gif
Output assets/demo.mp4

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 720
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

# Intro
Type "echo '⚡ paw-kit: reliability harness for Program-as-Weights functions (mock-backend walkthrough)'"
Enter
Sleep 1.5s

# 1. Instant Zero-Hardware Demo
Type "# Step 1: Instant zero-hardware demo (zero GPU, zero API keys)"
Enter
Sleep 1s

Type "uv run paw-kit demo"
Enter
Sleep 5s

# 2. JIT Tracing and Hot-Swapping Demo
Type "# Step 2: Automatic SQLite tracing and JIT hot-swapping (<1ms local execution)"
Enter
Sleep 1s

Type "uv run python examples/triage_ticket/run.py"
Enter
Sleep 6s

# 3. Test-Driven Neural Hardening with paw-test
Type "# Step 3: Test-driven neural hardening with adversarial fuzzing & auto-repair"
Enter
Sleep 1s

Type "uv run paw-test check examples/date_normalizer/suite.yaml"
Enter
Sleep 4s

# 4. Inspect Compiled Adapter
Type "# Step 4: Inspect compiled .paw neural adapter artifact"
Enter
Sleep 1s

Type "uv run paw-inspect examples/date_normalizer/.paw_demo_dates/date_normalizer.paw"
Enter
Sleep 4s

# Outro
Type "echo '🚀 Simulated local adapter demo — MockPAWBackend, no real model, no measured benchmark.'"
Enter
Sleep 2s
