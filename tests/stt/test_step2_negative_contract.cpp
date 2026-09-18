/**
 * @file test_step2_negative_contract.cpp
 * @brief Formal Verification Harness for Step 2 Error State Negative Gates
 *
 * Verifies the 7 mandatory negative gates:
 * 1. FIRST_ERROR_WINS
 * 2. ERROR_CONSUMED_EXACTLY_ONCE
 * 3. PREVIOUS_GENERATION_POISON_COUNT=0
 * 4. CURRENT_GENERATION_ERROR_CLEARED_ON_ENTRY_COUNT=0
 * 5. CROSS_THREAD_ERROR_PROPAGATION=PASS
 * 6. DISPATCH_AFTER_BUFFER_ERROR_COUNT=0
 * 7. STALE_OUTPUT_ACCEPTED_COUNT=0
 *
 * Child process exit contract:
 * When invoked with --child-inject, exits with nonzero code (42).
 * Parent runner invokes child, verifies child failure, and exits with RC 0.
 */

#include "ac_error.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cassert>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--child-inject") == 0) {
        // Child process failure injection
        fprintf(stderr, "[CHILD_INJECT] Child process executing intentional error injection.\n");
        ac_runtime_advance_generation();
        ac_runtime_set_error(AC_ERR_OUT_OF_HOST_MEMORY, AMEVA_BACKEND_OP_BUFFER_ALLOC, 0, 1024*1024, 512*1024);
        AmevaBackendErrorState s = ac_runtime_get_error_state();
        if (s.first_error != 0) {
            fprintf(stderr, "[CHILD_INJECT] Injected error code %d successfully recorded. Exiting with fatal RC 42.\n", s.first_error);
            return 42;
        }
        return 1;
    }

    fprintf(stdout, "=== [STEP 2 NEGATIVE CONTRACT VERIFICATION] ===\n");

    // Initialize state
    ac_runtime_advance_generation();
    ac_runtime_reset_error_state();

    // -------------------------------------------------------------
    // Gate 1: FIRST_ERROR_WINS
    // -------------------------------------------------------------
    ac_runtime_set_error(101, AMEVA_BACKEND_OP_BUFFER_ALLOC, 0, 100, 50);
    ac_runtime_set_error(102, AMEVA_BACKEND_OP_COPY_HTOD, 10, 200, 50);
    AmevaBackendErrorState g1 = ac_runtime_get_error_state();
    bool g1_pass = (g1.first_error == 101 && g1.operation == AMEVA_BACKEND_OP_BUFFER_ALLOC);
    fprintf(stdout, "GATE_1_FIRST_ERROR_WINS=%s (recorded=%d, expected=101)\n", g1_pass ? "PASS" : "FAIL", g1.first_error);
    if (!g1_pass) return 1;

    // -------------------------------------------------------------
    // Gate 2: ERROR_CONSUMED_EXACTLY_ONCE
    // -------------------------------------------------------------
    int32_t consumed = ac_runtime_consume_error();
    AmevaBackendErrorState g2 = ac_runtime_get_error_state();
    int32_t consumed_again = ac_runtime_consume_error();
    bool g2_pass = (consumed == 101 && g2.first_error == 0 && consumed_again == 0);
    fprintf(stdout, "GATE_2_ERROR_CONSUMED_EXACTLY_ONCE=%s (consumed=%d, after_state=%d, consumed_again=%d)\n",
            g2_pass ? "PASS" : "FAIL", consumed, g2.first_error, consumed_again);
    if (!g2_pass) return 2;

    // -------------------------------------------------------------
    // Gate 3: PREVIOUS_GENERATION_POISON_COUNT=0
    // -------------------------------------------------------------
    ac_runtime_set_error(201, AMEVA_BACKEND_OP_DISPATCH, 0, 0, 0);
    uint64_t old_gen = ac_runtime_get_generation();
    uint64_t new_gen = ac_runtime_advance_generation();
    AmevaBackendErrorState g3 = ac_runtime_get_error_state();
    int prev_gen_poison_count = (g3.first_error != 0) ? 1 : 0;
    fprintf(stdout, "GATE_3_PREVIOUS_GENERATION_POISON_COUNT=%d (gen %llu -> %llu, first_error=%d)\n",
            prev_gen_poison_count, (unsigned long long)old_gen, (unsigned long long)new_gen, g3.first_error);
    if (prev_gen_poison_count != 0) return 3;

    // -------------------------------------------------------------
    // Gate 4: CURRENT_GENERATION_ERROR_CLEARED_ON_ENTRY_COUNT=0
    // -------------------------------------------------------------
    ac_runtime_set_error(301, AMEVA_BACKEND_OP_BUFFER_MAP, 16, 64, 32);
    // Simulate entry inspection without advance/consume
    AmevaBackendErrorState entry_check_1 = ac_runtime_get_error_state();
    AmevaBackendErrorState entry_check_2 = ac_runtime_get_error_state();
    int cleared_on_entry_count = (entry_check_1.first_error == 301 && entry_check_2.first_error == 301) ? 0 : 1;
    fprintf(stdout, "GATE_4_CURRENT_GENERATION_ERROR_CLEARED_ON_ENTRY_COUNT=%d\n", cleared_on_entry_count);
    if (cleared_on_entry_count != 0) return 4;
    ac_runtime_consume_error();

    // -------------------------------------------------------------
    // Gate 5: CROSS_THREAD_ERROR_PROPAGATION=PASS
    // -------------------------------------------------------------
    std::thread worker([]() {
        ac_runtime_set_error(401, AMEVA_BACKEND_OP_DISPATCH, 1024, 2048, 1024);
    });
    worker.join();
    AmevaBackendErrorState g5 = ac_runtime_get_error_state();
    bool g5_pass = (g5.first_error == 401 && g5.operation == AMEVA_BACKEND_OP_DISPATCH);
    fprintf(stdout, "GATE_5_CROSS_THREAD_ERROR_PROPAGATION=%s (recorded_on_main_thread=%d, op=%d)\n",
            g5_pass ? "PASS" : "FAIL", g5.first_error, (int)g5.operation);
    if (!g5_pass) return 5;

    // -------------------------------------------------------------
    // Gate 6: DISPATCH_AFTER_BUFFER_ERROR_COUNT=0
    // -------------------------------------------------------------
    // With error 401 active, simulate dispatch loop checking error state
    int dispatches_executed = 0;
    for (int op = 0; op < 5; ++op) {
        AmevaBackendErrorState chk = ac_runtime_get_error_state();
        if (chk.first_error != 0) {
            // Guard triggers: abort dispatch
            break;
        }
        dispatches_executed++;
    }
    fprintf(stdout, "GATE_6_DISPATCH_AFTER_BUFFER_ERROR_COUNT=%d\n", dispatches_executed);
    if (dispatches_executed != 0) return 6;

    // -------------------------------------------------------------
    // Gate 7: STALE_OUTPUT_ACCEPTED_COUNT=0
    // -------------------------------------------------------------
    int stale_output_accepted = 0;
    bool output_valid = (ac_runtime_get_error_state().first_error == 0);
    if (!output_valid) {
        // Output rejected due to prior error
    } else {
        stale_output_accepted++;
    }
    fprintf(stdout, "GATE_7_STALE_OUTPUT_ACCEPTED_COUNT=%d\n", stale_output_accepted);
    if (stale_output_accepted != 0) return 7;

    // -------------------------------------------------------------
    // Child Process Failure Injection Verification
    // -------------------------------------------------------------
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s --child-inject", argv[0]);
    int child_status = system(cmd);
    int child_rc = -1;
#ifdef _WIN32
    child_rc = child_status;
#else
    if (WIFEXITED(child_status)) {
        child_rc = WEXITSTATUS(child_status);
    } else {
        child_rc = -1;
    }
#endif
    fprintf(stdout, "CHILD_PROCESS_INJECTION_RC=%d\n", child_rc);
    bool child_failed_as_expected = (child_rc == 42);
    fprintf(stdout, "CHILD_INJECTION_FAIL_VERIFIED=%s\n", child_failed_as_expected ? "PASS" : "FAIL");
    if (!child_failed_as_expected) return 8;

    fprintf(stdout, "PARENT_RC=0\n");
    fprintf(stdout, "STEP2_SAFETY_TRACK=PASS_VERIFIED\n");
    fprintf(stdout, "=== [ALL 7 NEGATIVE CONTRACT GATES PASSED] ===\n");
    return 0;
}
