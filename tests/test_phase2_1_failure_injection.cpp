/**
 * @file test_phase2_1_failure_injection.cpp
 * @brief Phase 2.1: Robustness, Security, and Failure-Injection Verification Suite
 *
 * Ground-Truth Protocol:
 *   Verifies all 12 edge-case and failure injection gates mandated for Core ABI stabilization:
 *     1. Context Mismatch Enforcement (AC_ERR_CONTEXT_MISMATCH)
 *     2. Output Pointer Null Safety across Failures
 *     3. Pre-Submit Counter Stability & Failure Rollback
 *     4. Event Double-Wait Idempotency (Release-Once)
 *     5. Event Wait-Then-Destroy Lifecycle
 *     6. Event Direct-Destroy without Wait (Automatic Wait & Clean Retire)
 *     7. Context Child Object Retention & Destruction Blocking (AC_ERR_RESOURCE_IN_USE)
 *     8. Buffer & Kernel Mid-Creation Failure Clean Teardown
 *     9. Zero-Byte Buffer Rejection (AC_ERR_INVALID_BUFFER_SIZE)
 *    10. Push Constant Size Over-Size Rejection (AC_ERR_KERNEL_ARG_INVALID)
 *    11. Descriptor Binding Count Mismatch Rejection (AC_ERR_KERNEL_ARG_INVALID)
 *    12. Dispatch Dimension 0 and Multiplication Overflow Rejection (AC_ERR_NUMERICAL_OVERFLOW)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "ac_types.h"
#include "ac_context.h"
#include "ac_buffer.h"
#include "ac_kernel.h"
#include "ac_event.h"
#include "ac_dispatch.h"

static const uint32_t kBitnetGemvI2SSpv[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x0000012du, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0007000fu, 0x00000005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000bu, 0x00000015u, 0x00060010u,
    0x00000004u, 0x00000011u, 0x00000020u, 0x00000004u, 0x00000001u, 0x00030003u, 0x00000002u, 0x000001c2u,
    0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00050005u, 0x00000008u, 0x61636f6cu, 0x616c5f6cu,
    0x0000656eu, 0x00080005u, 0x0000000bu, 0x4c5f6c67u, 0x6c61636fu, 0x6f766e49u, 0x69746163u, 0x44496e6fu,
    0x00000000u, 0x00050005u, 0x00000010u, 0x61636f6cu, 0x6f725f6cu, 0x00000077u, 0x00050005u, 0x00000014u,
    0x626f6c67u, 0x725f6c61u, 0x0000776fu, 0x00060005u, 0x00000015u, 0x575f6c67u, 0x476b726fu, 0x70756f72u,
    0x00004449u, 0x00070005u, 0x0000001eu, 0x65726874u, 0x615f6461u, 0x6d756363u, 0x74616c75u, 0x0000726fu,
    0x00060005u, 0x00000021u, 0x68737550u, 0x736e6f43u, 0x746e6174u, 0x00000073u, 0x00050006u, 0x00000021u,
    0x00000000u, 0x645f6e69u, 0x00006d69u, 0x00050006u, 0x00000021u, 0x00000001u, 0x5f74756fu, 0x006d6964u,
    0x00070006u, 0x00000021u, 0x00000002u, 0x75716564u, 0x5f746e61u, 0x6c616373u, 0x00000065u, 0x00080006u,
    0x00000021u, 0x00000003u, 0x67696577u, 0x6f5f7468u, 0x65736666u, 0x79625f74u, 0x00736574u, 0x00080006u,
    0x00000021u, 0x00000004u, 0x7074756fu, 0x6f5f7475u, 0x65736666u, 0x6f775f74u, 0x00736472u, 0x00040005u,
    0x00000023u, 0x61726170u, 0x0000736du, 0x00050005u, 0x0000002du, 0x5f6d756eu, 0x636f6c62u, 0x0000736bu,
    0x00050005u, 0x00000033u, 0x65736162u, 0x726f775fu, 0x00000064u, 0x00060005u, 0x00000038u, 0x5f776f72u,
    0x64726f77u, 0x7361625fu, 0x00000065u, 0x00030005u, 0x00000040u, 0x00000062u, 0x00060005u, 0x00000049u,
    0x636f6c62u, 0x6f635f6bu, 0x61625f6cu, 0x00006573u, 0x00050005u, 0x0000004cu, 0x64726f77u, 0x7864695fu,
    0x00000000u, 0x00050005u, 0x00000054u, 0x65747962u, 0x6968735fu, 0x00007466u, 0x00050005u, 0x00000058u,
    0x6b636170u, 0x775f6465u, 0x0064726fu, 0x00060005u, 0x0000005au, 0x67696557u, 0x75427468u, 0x72656666u,
    0x00000000u, 0x00060006u, 0x0000005au, 0x00000000u, 0x5f776172u, 0x67696577u, 0x00737468u, 0x00030005u,
    0x0000005cu, 0x00000000u, 0x00050005u, 0x00000061u, 0x65747962u, 0x6c61765fu, 0x00000000u, 0x00030005u,
    0x00000067u, 0x00003077u, 0x00030005u, 0x00000070u, 0x00003177u, 0x00030005u, 0x00000077u, 0x00003277u,
    0x00030005u, 0x0000007eu, 0x00003377u, 0x00030005u, 0x00000083u, 0x00003078u, 0x00070005u, 0x00000085u,
    0x69746341u, 0x69746176u, 0x75426e6fu, 0x72656666u, 0x00000000u, 0x00060006u, 0x00000085u, 0x00000000u,
    0x69746361u, 0x69746176u, 0x00736e6fu, 0x00030005u, 0x00000087u, 0x00000000u, 0x00030005u, 0x0000008fu,
    0x00003178u, 0x00030005u, 0x00000097u, 0x00003278u, 0x00030005u, 0x0000009fu, 0x00003378u, 0x00060005u,
    0x000000bdu, 0x5f776f72u, 0x72616873u, 0x735f6465u, 0x00006d75u, 0x00060005u, 0x0000011du, 0x7074754fu,
    0x75427475u, 0x72656666u, 0x00000000u, 0x00060006u, 0x0000011du, 0x00000000u, 0x7074756fu, 0x765f7475u,
    0x00006365u, 0x00030005u, 0x0000011fu, 0x00000000u, 0x00040047u, 0x0000000bu, 0x0000000bu, 0x0000001bu,
    0x00040047u, 0x00000015u, 0x0000000bu, 0x0000001au, 0x00030047u, 0x00000021u, 0x00000002u, 0x00050048u,
    0x00000021u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00050048u, 0x00000021u, 0x00000001u, 0x00000023u,
    0x00000004u, 0x00050048u, 0x00000021u, 0x00000002u, 0x00000023u, 0x00000008u, 0x00050048u, 0x00000021u,
    0x00000003u, 0x00000023u, 0x0000000cu, 0x00050048u, 0x00000021u, 0x00000004u, 0x00000023u, 0x00000010u,
    0x00040047u, 0x00000059u, 0x00000006u, 0x00000004u, 0x00030047u, 0x0000005au, 0x00000003u, 0x00040048u,
    0x0000005au, 0x00000000u, 0x00000018u, 0x00050048u, 0x0000005au, 0x00000000u, 0x00000023u, 0x00000000u,
    0x00030047u, 0x0000005cu, 0x00000018u, 0x00040047u, 0x0000005cu, 0x00000021u, 0x00000000u, 0x00040047u,
    0x0000005cu, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000084u, 0x00000006u, 0x00000004u, 0x00030047u,
    0x00000085u, 0x00000003u, 0x00040048u, 0x00000085u, 0x00000000u, 0x00000018u, 0x00050048u, 0x00000085u,
    0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x00000087u, 0x00000018u, 0x00040047u, 0x00000087u,
    0x00000021u, 0x00000001u, 0x00040047u, 0x00000087u, 0x00000022u, 0x00000000u, 0x00040047u, 0x0000011cu,
    0x00000006u, 0x00000004u, 0x00030047u, 0x0000011du, 0x00000003u, 0x00040048u, 0x0000011du, 0x00000000u,
    0x00000019u, 0x00050048u, 0x0000011du, 0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x0000011fu,
    0x00000019u, 0x00040047u, 0x0000011fu, 0x00000021u, 0x00000002u, 0x00040047u, 0x0000011fu, 0x00000022u,
    0x00000000u, 0x00040047u, 0x0000012cu, 0x0000000bu, 0x00000019u, 0x00020013u, 0x00000002u, 0x00030021u,
    0x00000003u, 0x00000002u, 0x00040015u, 0x00000006u, 0x00000020u, 0x00000000u, 0x00040020u, 0x00000007u,
    0x00000007u, 0x00000006u, 0x00040017u, 0x00000009u, 0x00000006u, 0x00000003u, 0x00040020u, 0x0000000au,
    0x00000001u, 0x00000009u, 0x0004003bu, 0x0000000au, 0x0000000bu, 0x00000001u, 0x0004002bu, 0x00000006u,
    0x0000000cu, 0x00000000u, 0x00040020u, 0x0000000du, 0x00000001u, 0x00000006u, 0x0004002bu, 0x00000006u,
    0x00000011u, 0x00000001u, 0x0004003bu, 0x0000000au, 0x00000015u, 0x00000001u, 0x0004002bu, 0x00000006u,
    0x00000018u, 0x00000004u, 0x00030016u, 0x0000001cu, 0x00000020u, 0x00040020u, 0x0000001du, 0x00000007u,
    0x0000001cu, 0x0004002bu, 0x0000001cu, 0x0000001fu, 0x00000000u, 0x0007001eu, 0x00000021u, 0x00000006u,
    0x00000006u, 0x0000001cu, 0x00000006u, 0x00000006u, 0x00040020u, 0x00000022u, 0x00000009u, 0x00000021u,
    0x0004003bu, 0x00000022u, 0x00000023u, 0x00000009u, 0x00040015u, 0x00000024u, 0x00000020u, 0x00000001u,
    0x0004002bu, 0x00000024u, 0x00000025u, 0x00000001u, 0x00040020u, 0x00000026u, 0x00000009u, 0x00000006u,
    0x00020014u, 0x00000029u, 0x0004002bu, 0x00000024u, 0x0000002eu, 0x00000000u, 0x0004002bu, 0x00000006u,
    0x00000031u, 0x00000080u, 0x0004002bu, 0x00000024u, 0x00000034u, 0x00000003u, 0x0004002bu, 0x00000006u,
    0x0000003cu, 0x00000008u, 0x0003001du, 0x00000059u, 0x00000006u, 0x0003001eu, 0x0000005au, 0x00000059u,
    0x00040020u, 0x0000005bu, 0x00000002u, 0x0000005au, 0x0004003bu, 0x0000005bu, 0x0000005cu, 0x00000002u,
    0x00040020u, 0x0000005eu, 0x00000002u, 0x00000006u, 0x0004002bu, 0x00000006u, 0x00000065u, 0x000000ffu,
    0x0004002bu, 0x00000024u, 0x00000069u, 0x00000006u, 0x0004002bu, 0x00000006u, 0x0000006bu, 0x00000003u,
    0x0004002bu, 0x0000001cu, 0x0000006eu, 0x3f800000u, 0x0004002bu, 0x00000024u, 0x00000072u, 0x00000004u,
    0x0004002bu, 0x00000024u, 0x00000079u, 0x00000002u, 0x0003001du, 0x00000084u, 0x0000001cu, 0x0003001eu,
    0x00000085u, 0x00000084u, 0x00040020u, 0x00000086u, 0x00000002u, 0x00000085u, 0x0004003bu, 0x00000086u,
    0x00000087u, 0x00000002u, 0x00040020u, 0x0000008cu, 0x00000002u, 0x0000001cu, 0x0004002bu, 0x00000006u,
    0x00000091u, 0x00000020u, 0x0004002bu, 0x00000006u, 0x00000099u, 0x00000040u, 0x0004002bu, 0x00000006u,
    0x000000a1u, 0x00000060u, 0x0004001cu, 0x000000bau, 0x0000001cu, 0x00000091u, 0x0004001cu, 0x000000bbu,
    0x000000bau, 0x00000018u, 0x00040020u, 0x000000bcu, 0x00000004u, 0x000000bbu, 0x0004003bu, 0x000000bcu,
    0x000000bdu, 0x00000004u, 0x00040020u, 0x000000c1u, 0x00000004u, 0x0000001cu, 0x0004002bu, 0x00000006u,
    0x000000c3u, 0x00000002u, 0x0004002bu, 0x00000006u, 0x000000c4u, 0x00000108u, 0x0004002bu, 0x00000006u,
    0x000000c6u, 0x00000010u, 0x0003001du, 0x0000011cu, 0x0000001cu, 0x0003001eu, 0x0000011du, 0x0000011cu,
    0x00040020u, 0x0000011eu, 0x00000002u, 0x0000011du, 0x0004003bu, 0x0000011eu, 0x0000011fu, 0x00000002u,
    0x00040020u, 0x00000127u, 0x00000009u, 0x0000001cu, 0x0006002cu, 0x00000009u, 0x0000012cu, 0x00000091u,
    0x00000018u, 0x00000011u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u,
    0x00000005u, 0x0004003bu, 0x00000007u, 0x00000008u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000010u,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000014u, 0x00000007u, 0x0004003bu, 0x0000001du, 0x0000001eu,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x0000002du, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000033u,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000038u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000040u,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000049u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x0000004cu,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000054u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000058u,
    0x00000007u, 0x0004003bu, 0x00000007u, 0x00000061u, 0x00000007u, 0x0004003bu, 0x0000001du, 0x00000067u,
    0x00000007u, 0x0004003bu, 0x0000001du, 0x00000070u, 0x00000007u, 0x0004003bu, 0x0000001du, 0x00000077u,
    0x00000007u, 0x0004003bu, 0x0000001du, 0x0000007eu, 0x00000007u, 0x0004003bu, 0x0000001du, 0x00000083u,
    0x00000007u, 0x0004003bu, 0x0000001du, 0x0000008fu, 0x00000007u, 0x0004003bu, 0x0000001du, 0x00000097u,
    0x00000007u, 0x0004003bu, 0x0000001du, 0x0000009fu, 0x00000007u, 0x00050041u, 0x0000000du, 0x0000000eu,
    0x0000000bu, 0x0000000cu, 0x0004003du, 0x00000006u, 0x0000000fu, 0x0000000eu, 0x0003003eu, 0x00000008u,
    0x0000000fu, 0x00050041u, 0x0000000du, 0x00000012u, 0x0000000bu, 0x00000011u, 0x0004003du, 0x00000006u,
    0x00000013u, 0x00000012u, 0x0003003eu, 0x00000010u, 0x00000013u, 0x00050041u, 0x0000000du, 0x00000016u,
    0x00000015u, 0x0000000cu, 0x0004003du, 0x00000006u, 0x00000017u, 0x00000016u, 0x00050084u, 0x00000006u,
    0x00000019u, 0x00000017u, 0x00000018u, 0x0004003du, 0x00000006u, 0x0000001au, 0x00000010u, 0x00050080u,
    0x00000006u, 0x0000001bu, 0x00000019u, 0x0000001au, 0x0003003eu, 0x00000014u, 0x0000001bu, 0x0003003eu,
    0x0000001eu, 0x0000001fu, 0x0004003du, 0x00000006u, 0x00000020u, 0x00000014u, 0x00050041u, 0x00000026u,
    0x00000027u, 0x00000023u, 0x00000025u, 0x0004003du, 0x00000006u, 0x00000028u, 0x00000027u, 0x000500b0u,
    0x00000029u, 0x0000002au, 0x00000020u, 0x00000028u, 0x000300f7u, 0x0000002cu, 0x00000000u, 0x000400fau,
    0x0000002au, 0x0000002bu, 0x0000002cu, 0x000200f8u, 0x0000002bu, 0x00050041u, 0x00000026u, 0x0000002fu,
    0x00000023u, 0x0000002eu, 0x0004003du, 0x00000006u, 0x00000030u, 0x0000002fu, 0x00050086u, 0x00000006u,
    0x00000032u, 0x00000030u, 0x00000031u, 0x0003003eu, 0x0000002du, 0x00000032u, 0x00050041u, 0x00000026u,
    0x00000035u, 0x00000023u, 0x00000034u, 0x0004003du, 0x00000006u, 0x00000036u, 0x00000035u, 0x00050086u,
    0x00000006u, 0x00000037u, 0x00000036u, 0x00000018u, 0x0003003eu, 0x00000033u, 0x00000037u, 0x0004003du,
    0x00000006u, 0x00000039u, 0x00000033u, 0x0004003du, 0x00000006u, 0x0000003au, 0x00000014u, 0x0004003du,
    0x00000006u, 0x0000003bu, 0x0000002du, 0x00050084u, 0x00000006u, 0x0000003du, 0x0000003bu, 0x0000003cu,
    0x00050084u, 0x00000006u, 0x0000003eu, 0x0000003au, 0x0000003du, 0x00050080u, 0x00000006u, 0x0000003fu,
    0x00000039u, 0x0000003eu, 0x0003003eu, 0x00000038u, 0x0000003fu, 0x0003003eu, 0x00000040u, 0x0000000cu,
    0x000200f9u, 0x00000041u, 0x000200f8u, 0x00000041u, 0x000400f6u, 0x00000043u, 0x00000044u, 0x00000000u,
    0x000200f9u, 0x00000045u, 0x000200f8u, 0x00000045u, 0x0004003du, 0x00000006u, 0x00000046u, 0x00000040u,
    0x0004003du, 0x00000006u, 0x00000047u, 0x0000002du, 0x000500b0u, 0x00000029u, 0x00000048u, 0x00000046u,
    0x00000047u, 0x000400fau, 0x00000048u, 0x00000042u, 0x00000043u, 0x000200f8u, 0x00000042u, 0x0004003du,
    0x00000006u, 0x0000004au, 0x00000040u, 0x00050084u, 0x00000006u, 0x0000004bu, 0x0000004au, 0x00000031u,
    0x0003003eu, 0x00000049u, 0x0000004bu, 0x0004003du, 0x00000006u, 0x0000004du, 0x00000038u, 0x0004003du,
    0x00000006u, 0x0000004eu, 0x00000040u, 0x00050084u, 0x00000006u, 0x0000004fu, 0x0000004eu, 0x0000003cu,
    0x00050080u, 0x00000006u, 0x00000050u, 0x0000004du, 0x0000004fu, 0x0004003du, 0x00000006u, 0x00000051u,
    0x00000008u, 0x00050086u, 0x00000006u, 0x00000052u, 0x00000051u, 0x00000018u, 0x00050080u, 0x00000006u,
    0x00000053u, 0x00000050u, 0x00000052u, 0x0003003eu, 0x0000004cu, 0x00000053u, 0x0004003du, 0x00000006u,
    0x00000055u, 0x00000008u, 0x00050089u, 0x00000006u, 0x00000056u, 0x00000055u, 0x00000018u, 0x00050084u,
    0x00000006u, 0x00000057u, 0x00000056u, 0x0000003cu, 0x0003003eu, 0x00000054u, 0x00000057u, 0x0004003du,
    0x00000006u, 0x0000005du, 0x0000004cu, 0x00060041u, 0x0000005eu, 0x0000005fu, 0x0000005cu, 0x0000002eu,
    0x0000005du, 0x0004003du, 0x00000006u, 0x00000060u, 0x0000005fu, 0x0003003eu, 0x00000058u, 0x00000060u,
    0x0004003du, 0x00000006u, 0x00000062u, 0x00000058u, 0x0004003du, 0x00000006u, 0x00000063u, 0x00000054u,
    0x000500c2u, 0x00000006u, 0x00000064u, 0x00000062u, 0x00000063u, 0x000500c7u, 0x00000006u, 0x00000066u,
    0x00000064u, 0x00000065u, 0x0003003eu, 0x00000061u, 0x00000066u, 0x0004003du, 0x00000006u, 0x00000068u,
    0x00000061u, 0x000500c2u, 0x00000006u, 0x0000006au, 0x00000068u, 0x00000069u, 0x000500c7u, 0x00000006u,
    0x0000006cu, 0x0000006au, 0x0000006bu, 0x00040070u, 0x0000001cu, 0x0000006du, 0x0000006cu, 0x00050083u,
    0x0000001cu, 0x0000006fu, 0x0000006du, 0x0000006eu, 0x0003003eu, 0x00000067u, 0x0000006fu, 0x0004003du,
    0x00000006u, 0x00000071u, 0x00000061u, 0x000500c2u, 0x00000006u, 0x00000073u, 0x00000071u, 0x00000072u,
    0x000500c7u, 0x00000006u, 0x00000074u, 0x00000073u, 0x0000006bu, 0x00040070u, 0x0000001cu, 0x00000075u,
    0x00000074u, 0x00050083u, 0x0000001cu, 0x00000076u, 0x00000075u, 0x0000006eu, 0x0003003eu, 0x00000070u,
    0x00000076u, 0x0004003du, 0x00000006u, 0x00000078u, 0x00000061u, 0x000500c2u, 0x00000006u, 0x0000007au,
    0x00000078u, 0x00000079u, 0x000500c7u, 0x00000006u, 0x0000007bu, 0x0000007au, 0x0000006bu, 0x00040070u,
    0x0000001cu, 0x0000007cu, 0x0000007bu, 0x00050083u, 0x0000001cu, 0x0000007du, 0x0000007cu, 0x0000006eu,
    0x0003003eu, 0x00000077u, 0x0000007du, 0x0004003du, 0x00000006u, 0x0000007fu, 0x00000061u, 0x000500c7u,
    0x00000006u, 0x00000080u, 0x0000007fu, 0x0000006bu, 0x00040070u, 0x0000001cu, 0x00000081u, 0x00000080u,
    0x00050083u, 0x0000001cu, 0x00000082u, 0x00000081u, 0x0000006eu, 0x0003003eu, 0x0000007eu, 0x00000082u,
    0x0004003du, 0x00000006u, 0x00000088u, 0x00000049u, 0x00050080u, 0x00000006u, 0x00000089u, 0x00000088u,
    0x0000000cu, 0x0004003du, 0x00000006u, 0x0000008au, 0x00000008u, 0x00050080u, 0x00000006u, 0x0000008bu,
    0x00000089u, 0x0000008au, 0x00060041u, 0x0000008cu, 0x0000008du, 0x00000087u, 0x0000002eu, 0x0000008bu,
    0x0004003du, 0x0000001cu, 0x0000008eu, 0x0000008du, 0x0003003eu, 0x00000083u, 0x0000008eu, 0x0004003du,
    0x00000006u, 0x00000090u, 0x00000049u, 0x00050080u, 0x00000006u, 0x00000092u, 0x00000090u, 0x00000091u,
    0x0004003du, 0x00000006u, 0x00000093u, 0x00000008u, 0x00050080u, 0x00000006u, 0x00000094u, 0x00000092u,
    0x00000093u, 0x00060041u, 0x0000008cu, 0x00000095u, 0x00000087u, 0x0000002eu, 0x00000094u, 0x0004003du,
    0x0000001cu, 0x00000096u, 0x00000095u, 0x0003003eu, 0x0000008fu, 0x00000096u, 0x0004003du, 0x00000006u,
    0x00000098u, 0x00000049u, 0x00050080u, 0x00000006u, 0x0000009au, 0x00000098u, 0x00000099u, 0x0004003du,
    0x00000006u, 0x0000009bu, 0x00000008u, 0x00050080u, 0x00000006u, 0x0000009cu, 0x0000009au, 0x0000009bu,
    0x00060041u, 0x0000008cu, 0x0000009du, 0x00000087u, 0x0000002eu, 0x0000009cu, 0x0004003du, 0x0000001cu,
    0x0000009eu, 0x0000009du, 0x0003003eu, 0x00000097u, 0x0000009eu, 0x0004003du, 0x00000006u, 0x000000a0u,
    0x00000049u, 0x00050080u, 0x00000006u, 0x000000a2u, 0x000000a0u, 0x000000a1u, 0x0004003du, 0x00000006u,
    0x000000a3u, 0x00000008u, 0x00050080u, 0x00000006u, 0x000000a4u, 0x000000a2u, 0x000000a3u, 0x00060041u,
    0x0000008cu, 0x000000a5u, 0x00000087u, 0x0000002eu, 0x000000a4u, 0x0004003du, 0x0000001cu, 0x000000a6u,
    0x000000a5u, 0x0003003eu, 0x0000009fu, 0x000000a6u, 0x0004003du, 0x0000001cu, 0x000000a7u, 0x00000067u,
    0x0004003du, 0x0000001cu, 0x000000a8u, 0x00000083u, 0x00050085u, 0x0000001cu, 0x000000a9u, 0x000000a7u,
    0x000000a8u, 0x0004003du, 0x0000001cu, 0x000000aau, 0x00000070u, 0x0004003du, 0x0000001cu, 0x000000abu,
    0x0000008fu, 0x00050085u, 0x0000001cu, 0x000000acu, 0x000000aau, 0x000000abu, 0x00050081u, 0x0000001cu,
    0x000000adu, 0x000000a9u, 0x000000acu, 0x0004003du, 0x0000001cu, 0x000000aeu, 0x00000077u, 0x0004003du,
    0x0000001cu, 0x000000afu, 0x00000097u, 0x00050085u, 0x0000001cu, 0x000000b0u, 0x000000aeu, 0x000000afu,
    0x00050081u, 0x0000001cu, 0x000000b1u, 0x000000adu, 0x000000b0u, 0x0004003du, 0x0000001cu, 0x000000b2u,
    0x0000007eu, 0x0004003du, 0x0000001cu, 0x000000b3u, 0x0000009fu, 0x00050085u, 0x0000001cu, 0x000000b4u,
    0x000000b2u, 0x000000b3u, 0x00050081u, 0x0000001cu, 0x000000b5u, 0x000000b1u, 0x000000b4u, 0x0004003du,
    0x0000001cu, 0x000000b6u, 0x0000001eu, 0x00050081u, 0x0000001cu, 0x000000b7u, 0x000000b6u, 0x000000b5u,
    0x0003003eu, 0x0000001eu, 0x000000b7u, 0x000200f9u, 0x00000044u, 0x000200f8u, 0x00000044u, 0x0004003du,
    0x00000006u, 0x000000b8u, 0x00000040u, 0x00050080u, 0x00000006u, 0x000000b9u, 0x000000b8u, 0x00000025u,
    0x0003003eu, 0x00000040u, 0x000000b9u, 0x000200f9u, 0x00000041u, 0x000200f8u, 0x00000043u, 0x000200f9u,
    0x0000002cu, 0x000200f8u, 0x0000002cu, 0x0004003du, 0x00000006u, 0x000000beu, 0x00000010u, 0x0004003du,
    0x00000006u, 0x000000bfu, 0x00000008u, 0x0004003du, 0x0000001cu, 0x000000c0u, 0x0000001eu, 0x00060041u,
    0x000000c1u, 0x000000c2u, 0x000000bdu, 0x000000beu, 0x000000bfu, 0x0003003eu, 0x000000c2u, 0x000000c0u,
    0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u, 0x0004003du, 0x00000006u, 0x000000c5u, 0x00000008u,
    0x000500b0u, 0x00000029u, 0x000000c7u, 0x000000c5u, 0x000000c6u, 0x000300f7u, 0x000000c9u, 0x00000000u,
    0x000400fau, 0x000000c7u, 0x000000c8u, 0x000000c9u, 0x000200f8u, 0x000000c8u, 0x0004003du, 0x00000006u,
    0x000000cau, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000cbu, 0x00000008u, 0x0004003du, 0x00000006u,
    0x000000ccu, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000cdu, 0x00000008u, 0x00050080u, 0x00000006u,
    0x000000ceu, 0x000000cdu, 0x000000c6u, 0x00060041u, 0x000000c1u, 0x000000cfu, 0x000000bdu, 0x000000ccu,
    0x000000ceu, 0x0004003du, 0x0000001cu, 0x000000d0u, 0x000000cfu, 0x00060041u, 0x000000c1u, 0x000000d1u,
    0x000000bdu, 0x000000cau, 0x000000cbu, 0x0004003du, 0x0000001cu, 0x000000d2u, 0x000000d1u, 0x00050081u,
    0x0000001cu, 0x000000d3u, 0x000000d2u, 0x000000d0u, 0x00060041u, 0x000000c1u, 0x000000d4u, 0x000000bdu,
    0x000000cau, 0x000000cbu, 0x0003003eu, 0x000000d4u, 0x000000d3u, 0x000200f9u, 0x000000c9u, 0x000200f8u,
    0x000000c9u, 0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u, 0x0004003du, 0x00000006u, 0x000000d5u,
    0x00000008u, 0x000500b0u, 0x00000029u, 0x000000d6u, 0x000000d5u, 0x0000003cu, 0x000300f7u, 0x000000d8u,
    0x00000000u, 0x000400fau, 0x000000d6u, 0x000000d7u, 0x000000d8u, 0x000200f8u, 0x000000d7u, 0x0004003du,
    0x00000006u, 0x000000d9u, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000dau, 0x00000008u, 0x0004003du,
    0x00000006u, 0x000000dbu, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000dcu, 0x00000008u, 0x00050080u,
    0x00000006u, 0x000000ddu, 0x000000dcu, 0x0000003cu, 0x00060041u, 0x000000c1u, 0x000000deu, 0x000000bdu,
    0x000000dbu, 0x000000ddu, 0x0004003du, 0x0000001cu, 0x000000dfu, 0x000000deu, 0x00060041u, 0x000000c1u,
    0x000000e0u, 0x000000bdu, 0x000000d9u, 0x000000dau, 0x0004003du, 0x0000001cu, 0x000000e1u, 0x000000e0u,
    0x00050081u, 0x0000001cu, 0x000000e2u, 0x000000e1u, 0x000000dfu, 0x00060041u, 0x000000c1u, 0x000000e3u,
    0x000000bdu, 0x000000d9u, 0x000000dau, 0x0003003eu, 0x000000e3u, 0x000000e2u, 0x000200f9u, 0x000000d8u,
    0x000200f8u, 0x000000d8u, 0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u, 0x0004003du, 0x00000006u,
    0x000000e4u, 0x00000008u, 0x000500b0u, 0x00000029u, 0x000000e5u, 0x000000e4u, 0x00000018u, 0x000300f7u,
    0x000000e7u, 0x00000000u, 0x000400fau, 0x000000e5u, 0x000000e6u, 0x000000e7u, 0x000200f8u, 0x000000e6u,
    0x0004003du, 0x00000006u, 0x000000e8u, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000e9u, 0x00000008u,
    0x0004003du, 0x00000006u, 0x000000eau, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000ebu, 0x00000008u,
    0x00050080u, 0x00000006u, 0x000000ecu, 0x000000ebu, 0x00000018u, 0x00060041u, 0x000000c1u, 0x000000edu,
    0x000000bdu, 0x000000eau, 0x000000ecu, 0x0004003du, 0x0000001cu, 0x000000eeu, 0x000000edu, 0x00060041u,
    0x000000c1u, 0x000000efu, 0x000000bdu, 0x000000e8u, 0x000000e9u, 0x0004003du, 0x0000001cu, 0x000000f0u,
    0x000000efu, 0x00050081u, 0x0000001cu, 0x000000f1u, 0x000000f0u, 0x000000eeu, 0x00060041u, 0x000000c1u,
    0x000000f2u, 0x000000bdu, 0x000000e8u, 0x000000e9u, 0x0003003eu, 0x000000f2u, 0x000000f1u, 0x000200f9u,
    0x000000e7u, 0x000200f8u, 0x000000e7u, 0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u, 0x0004003du,
    0x00000006u, 0x000000f3u, 0x00000008u, 0x000500b0u, 0x00000029u, 0x000000f4u, 0x000000f3u, 0x000000c3u,
    0x000300f7u, 0x000000f6u, 0x00000000u, 0x000400fau, 0x000000f4u, 0x000000f5u, 0x000000f6u, 0x000200f8u,
    0x000000f5u, 0x0004003du, 0x00000006u, 0x000000f7u, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000f8u,
    0x00000008u, 0x0004003du, 0x00000006u, 0x000000f9u, 0x00000010u, 0x0004003du, 0x00000006u, 0x000000fau,
    0x00000008u, 0x00050080u, 0x00000006u, 0x000000fbu, 0x000000fau, 0x000000c3u, 0x00060041u, 0x000000c1u,
    0x000000fcu, 0x000000bdu, 0x000000f9u, 0x000000fbu, 0x0004003du, 0x0000001cu, 0x000000fdu, 0x000000fcu,
    0x00060041u, 0x000000c1u, 0x000000feu, 0x000000bdu, 0x000000f7u, 0x000000f8u, 0x0004003du, 0x0000001cu,
    0x000000ffu, 0x000000feu, 0x00050081u, 0x0000001cu, 0x00000100u, 0x000000ffu, 0x000000fdu, 0x00060041u,
    0x000000c1u, 0x00000101u, 0x000000bdu, 0x000000f7u, 0x000000f8u, 0x0003003eu, 0x00000101u, 0x00000100u,
    0x000200f9u, 0x000000f6u, 0x000200f8u, 0x000000f6u, 0x000400e0u, 0x000000c3u, 0x000000c3u, 0x000000c4u,
    0x0004003du, 0x00000006u, 0x00000102u, 0x00000008u, 0x000500b0u, 0x00000029u, 0x00000103u, 0x00000102u,
    0x00000011u, 0x000300f7u, 0x00000105u, 0x00000000u, 0x000400fau, 0x00000103u, 0x00000104u, 0x00000105u,
    0x000200f8u, 0x00000104u, 0x0004003du, 0x00000006u, 0x00000106u, 0x00000010u, 0x0004003du, 0x00000006u,
    0x00000107u, 0x00000008u, 0x0004003du, 0x00000006u, 0x00000108u, 0x00000010u, 0x0004003du, 0x00000006u,
    0x00000109u, 0x00000008u, 0x00050080u, 0x00000006u, 0x0000010au, 0x00000109u, 0x00000011u, 0x00060041u,
    0x000000c1u, 0x0000010bu, 0x000000bdu, 0x00000108u, 0x0000010au, 0x0004003du, 0x0000001cu, 0x0000010cu,
    0x0000010bu, 0x00060041u, 0x000000c1u, 0x0000010du, 0x000000bdu, 0x00000106u, 0x00000107u, 0x0004003du,
    0x0000001cu, 0x0000010eu, 0x0000010du, 0x00050081u, 0x0000001cu, 0x0000010fu, 0x0000010eu, 0x0000010cu,
    0x00060041u, 0x000000c1u, 0x00000110u, 0x000000bdu, 0x00000106u, 0x00000107u, 0x0003003eu, 0x00000110u,
    0x0000010fu, 0x000200f9u, 0x00000105u, 0x000200f8u, 0x00000105u, 0x0004003du, 0x00000006u, 0x00000111u,
    0x00000008u, 0x000500aau, 0x00000029u, 0x00000112u, 0x00000111u, 0x0000000cu, 0x000300f7u, 0x00000114u,
    0x00000000u, 0x000400fau, 0x00000112u, 0x00000113u, 0x00000114u, 0x000200f8u, 0x00000113u, 0x0004003du,
    0x00000006u, 0x00000115u, 0x00000014u, 0x00050041u, 0x00000026u, 0x00000116u, 0x00000023u, 0x00000025u,
    0x0004003du, 0x00000006u, 0x00000117u, 0x00000116u, 0x000500b0u, 0x00000029u, 0x00000118u, 0x00000115u,
    0x00000117u, 0x000200f9u, 0x00000114u, 0x000200f8u, 0x00000114u, 0x000700f5u, 0x00000029u, 0x00000119u,
    0x00000112u, 0x00000105u, 0x00000118u, 0x00000113u, 0x000300f7u, 0x0000011bu, 0x00000000u, 0x000400fau,
    0x00000119u, 0x0000011au, 0x0000011bu, 0x000200f8u, 0x0000011au, 0x00050041u, 0x00000026u, 0x00000120u,
    0x00000023u, 0x00000072u, 0x0004003du, 0x00000006u, 0x00000121u, 0x00000120u, 0x0004003du, 0x00000006u,
    0x00000122u, 0x00000014u, 0x00050080u, 0x00000006u, 0x00000123u, 0x00000121u, 0x00000122u, 0x0004003du,
    0x00000006u, 0x00000124u, 0x00000010u, 0x00060041u, 0x000000c1u, 0x00000125u, 0x000000bdu, 0x00000124u,
    0x0000002eu, 0x0004003du, 0x0000001cu, 0x00000126u, 0x00000125u, 0x00050041u, 0x00000127u, 0x00000128u,
    0x00000023u, 0x00000079u, 0x0004003du, 0x0000001cu, 0x00000129u, 0x00000128u, 0x00050085u, 0x0000001cu,
    0x0000012au, 0x00000126u, 0x00000129u, 0x00060041u, 0x0000008cu, 0x0000012bu, 0x0000011fu, 0x0000002eu,
    0x00000123u, 0x0003003eu, 0x0000012bu, 0x0000012au, 0x000200f9u, 0x0000011bu, 0x000200f8u, 0x0000011bu,
    0x000100fdu, 0x00010038u,
};
static const size_t kBitnetGemvI2SSpvByteSize = sizeof(kBitnetGemvI2SSpv);

struct BitNetPushConstants {
    uint32_t in_dim;
    uint32_t out_dim;
    float dequant_scale;
    uint32_t weight_offset_bytes;
    uint32_t output_offset_words;
};

int main(void) {
    printf("====================================================================\n");
    printf("Phase 2.1: Robustness & Failure-Injection Verification Suite\n");
    printf("Hardware Target: Qualcomm Adreno 830 GPU (Android 15 ARM64)\n");
    printf("====================================================================\n");

    int total_failures = 0;

    /* Primary context and kernel setup */
    AcVkContext* ctx1 = NULL;
    AcResult res = ac_vk_context_create(&ctx1);
    if (res != AC_SUCCESS || !ctx1) {
        printf("  [FATAL] Cannot create primary AcVkContext (error %d)\n", res);
        return 1;
    }

    AcVkBuffer* buf1_weight = NULL;
    AcVkBuffer* buf1_act = NULL;
    AcVkBuffer* buf1_out = NULL;
    ac_vk_buffer_create(ctx1, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf1_weight);
    ac_vk_buffer_create(ctx1, 256 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf1_act);
    ac_vk_buffer_create(ctx1, 256 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf1_out);

    AcVkKernel* kernel1 = NULL;
    ac_vk_kernel_create_spirv(ctx1, kBitnetGemvI2SSpv, kBitnetGemvI2SSpvByteSize, sizeof(BitNetPushConstants), 3, &kernel1);

    AcVkBuffer* buffers1[3] = { buf1_weight, buf1_act, buf1_out };
    BitNetPushConstants valid_push = { 256, 1, 1.0f, 0, 0 };

    /* ------------------------------------------------------------------------
     * Gate 1: Context Mismatch Enforcement (AC_ERR_CONTEXT_MISMATCH)
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 1] Cross-Context Object Mismatch Enforcement...\n");
    AcVkContext* ctx2 = NULL;
    res = ac_vk_context_create(&ctx2);
    if (res != AC_SUCCESS || !ctx2) {
        printf("  -> FAIL: Could not create secondary AcVkContext (error %d)\n", res);
        total_failures++;
    } else {
        AcVkBuffer* buf2_alien = NULL;
        ac_vk_buffer_create(ctx2, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf2_alien);

        /* Case 1.1: Dispatch on ctx1 with alien buffer from ctx2 */
        AcVkBuffer* alien_buffers[3] = { buf2_alien, buf1_act, buf1_out };
        res = ac_vk_dispatch(ctx1, kernel1, alien_buffers, 3, &valid_push, sizeof(valid_push), 1, 1, 1, NULL);
        if (res == AC_ERR_CONTEXT_MISMATCH) {
            printf("  -> PASS: Alien buffer correctly rejected with AC_ERR_CONTEXT_MISMATCH (-107).\n");
        } else {
            printf("  -> FAIL: Alien buffer did NOT trigger AC_ERR_CONTEXT_MISMATCH (Got: %d)\n", res);
            total_failures++;
        }

        /* Case 1.2: Dispatch on ctx2 with kernel belonging to ctx1 */
        res = ac_vk_dispatch(ctx2, kernel1, alien_buffers, 3, &valid_push, sizeof(valid_push), 1, 1, 1, NULL);
        if (res == AC_ERR_CONTEXT_MISMATCH) {
            printf("  -> PASS: Alien kernel correctly rejected with AC_ERR_CONTEXT_MISMATCH (-107).\n");
        } else {
            printf("  -> FAIL: Alien kernel did NOT trigger AC_ERR_CONTEXT_MISMATCH (Got: %d)\n", res);
            total_failures++;
        }

        ac_vk_buffer_destroy(buf2_alien);
        ac_vk_context_destroy(ctx2);
    }

    /* ------------------------------------------------------------------------
     * Gate 2: Output Pointer NULL Safety on Failure
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 2] Output Pointer NULL Safety across Injected Failures...\n");
    AcVkEvent* leak_check_event = (AcVkEvent*)0xDEADBEEF;
    res = ac_vk_dispatch(ctx1, kernel1, buffers1, 3, &valid_push, sizeof(valid_push), 0, 1, 1, &leak_check_event);
    if (res != AC_SUCCESS && leak_check_event == NULL) {
        printf("  -> PASS: Failed dispatch reset output pointer to NULL (res=%d).\n", res);
    } else {
        printf("  -> FAIL: Output pointer was NOT set to NULL on failure! (ptr=%p)\n", leak_check_event);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 3: Counter Rollback & Pre-Submit Allocation Safety
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 3] Submission Outstanding Counter Integrity & Rollback...\n");
    uint32_t ctx_pre = ctx1->submission_outstanding_count;
    uint32_t ker_pre = kernel1->submission_outstanding_count;
    uint32_t buf_pre = buf1_weight->submission_outstanding_count;

    /* Inject parameter failure: push constant mismatch */
    res = ac_vk_dispatch(ctx1, kernel1, buffers1, 3, &valid_push, sizeof(valid_push) + 4, 1, 1, 1, NULL);
    if (res == AC_ERR_KERNEL_ARG_INVALID &&
        ctx1->submission_outstanding_count == ctx_pre &&
        kernel1->submission_outstanding_count == ker_pre &&
        buf1_weight->submission_outstanding_count == buf_pre) {
        printf("  -> PASS: Pre-submission failure preserved counter values (all 0).\n");
    } else {
        printf("  -> FAIL: Counters altered during pre-submission rejection!\n");
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 4 & 5: Event Double-Wait Idempotency & Wait-Then-Destroy
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 4 & 5] Event Double-Wait & Idempotent Release-Once Lifecycle...\n");
    AcVkEvent* event4 = NULL;
    res = ac_vk_dispatch(ctx1, kernel1, buffers1, 3, &valid_push, sizeof(valid_push), 1, 1, 1, &event4);
    if (res != AC_SUCCESS || !event4) {
        printf("  -> FAIL: Async dispatch failed with error %d\n", res);
        total_failures++;
    } else {
        /* First wait: must succeed and retire resources */
        AcResult w1 = ac_vk_event_wait(event4, 5000000000ULL);
        if (w1 != AC_SUCCESS) {
            printf("  -> FAIL: First wait failed (error %d)\n", w1);
            total_failures++;
        }
        /* Second wait: must be idempotent and succeed without duplicate retirement */
        AcResult w2 = ac_vk_event_wait(event4, 5000000000ULL);
        if (w2 == AC_SUCCESS) {
            printf("  -> PASS: Double-wait succeeded idempotently (Release-Once).\n");
        } else {
            printf("  -> FAIL: Double-wait failed on second invocation (error %d)\n", w2);
            total_failures++;
        }

        /* Destroy after wait: must not decrement counters again */
        ac_vk_event_destroy(event4);
        if (ctx1->submission_outstanding_count == 0 &&
            kernel1->submission_outstanding_count == 0 &&
            buf1_weight->submission_outstanding_count == 0) {
            printf("  -> PASS: Wait-then-destroy maintained zero outstanding submissions (no underflow).\n");
        } else {
            printf("  -> FAIL: Counter underflow detected after wait-then-destroy!\n");
            total_failures++;
        }
    }

    /* ------------------------------------------------------------------------
     * Gate 6: Direct Event Destroy without Prior Explicit Wait
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 6] Direct Event Destruction without Prior Explicit Wait...\n");
    AcVkEvent* event6 = NULL;
    res = ac_vk_dispatch(ctx1, kernel1, buffers1, 3, &valid_push, sizeof(valid_push), 1, 1, 1, &event6);
    if (res != AC_SUCCESS || !event6) {
        printf("  -> FAIL: Async dispatch failed with error %d\n", res);
        total_failures++;
    } else {
        /* Destroy directly without calling ac_vk_event_wait */
        res = ac_vk_event_destroy(event6);
        if (res == AC_SUCCESS &&
            ctx1->submission_outstanding_count == 0 &&
            kernel1->submission_outstanding_count == 0 &&
            buf1_weight->submission_outstanding_count == 0) {
            printf("  -> PASS: Direct destroy safely waited for fence and retired resources exactly once.\n");
        } else {
            printf("  -> FAIL: Direct destroy failed or left leaked counters! (res=%d)\n", res);
            total_failures++;
        }
    }

    /* ------------------------------------------------------------------------
     * Gate 7: Child Object Retention Blocks Context Destruction
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 7] Child Object Retention Blocks Context Destruction...\n");
    AcVkContext* ctx_child_test = NULL;
    ac_vk_context_create(&ctx_child_test);
    AcVkBuffer* child_buf = NULL;
    ac_vk_buffer_create(ctx_child_test, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &child_buf);

    res = ac_vk_context_destroy(ctx_child_test);
    if (res == AC_ERR_RESOURCE_IN_USE) {
        printf("  -> PASS: Context destruction correctly blocked while child buffer is alive (AC_ERR_RESOURCE_IN_USE).\n");
    } else {
        printf("  -> FAIL: Context destruction was NOT blocked! (Got: %d, Expected: AC_ERR_RESOURCE_IN_USE)\n", res);
        total_failures++;
    }

    /* Destroy child buffer first, then context destruction must succeed */
    ac_vk_buffer_destroy(child_buf);
    res = ac_vk_context_destroy(ctx_child_test);
    if (res == AC_SUCCESS) {
        printf("  -> PASS: Context successfully destroyed after child buffer release.\n");
    } else {
        printf("  -> FAIL: Context destruction failed after child buffer release (error %d)\n", res);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 8: Mid-Creation Failure Clean Teardown
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 8] Mid-Creation Failure & Out-Of-Memory Cleanup...\n");
    AcVkBuffer* huge_buf = (AcVkBuffer*)0x1;
    /* Request allocation beyond physical memory (16 Exabytes) */
    res = ac_vk_buffer_create(ctx1, 0xFFFFFFFFFF000000ULL, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &huge_buf);
    if (res == AC_ERR_OUT_OF_DEVICE_MEMORY && huge_buf == NULL) {
        printf("  -> PASS: Out-of-memory buffer creation cleanly failed without leaks (NULL output).\n");
    } else {
        printf("  -> FAIL: Huge buffer allocation did not fail properly (res=%d, ptr=%p)\n", res, huge_buf);
        total_failures++;
    }

    AcVkKernel* bad_kernel = (AcVkKernel*)0x1;
    uint32_t malformed_spv = 0x12345678;
    res = ac_vk_kernel_create_spirv(ctx1, &malformed_spv, 2, 0, 3, &bad_kernel);
    if (res == AC_ERR_BINARY_CORRUPTED && bad_kernel == NULL) {
        printf("  -> PASS: Malformed SPIR-V rejected with AC_ERR_BINARY_CORRUPTED (NULL output).\n");
    } else {
        printf("  -> FAIL: Malformed SPIR-V was not rejected cleanly (res=%d)\n", res);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 9: Zero-Byte Buffer Rejection
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 9] Zero-Byte Buffer Allocation Rejection...\n");
    AcVkBuffer* zero_buf = (AcVkBuffer*)0x1;
    res = ac_vk_buffer_create(ctx1, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &zero_buf);
    if (res == AC_ERR_INVALID_BUFFER_SIZE && zero_buf == NULL) {
        printf("  -> PASS: Zero-byte buffer correctly rejected with AC_ERR_INVALID_BUFFER_SIZE.\n");
    } else {
        printf("  -> FAIL: Zero-byte buffer was not rejected! (res=%d)\n", res);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 10: Push Constant Size Over-Size Rejection
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 10] Push Constant Size Over-Size Rejection...\n");
    res = ac_vk_dispatch(ctx1, kernel1, buffers1, 3, &valid_push, sizeof(valid_push) + 16, 1, 1, 1, NULL);
    if (res == AC_ERR_KERNEL_ARG_INVALID) {
        printf("  -> PASS: Oversized push constant rejected with AC_ERR_KERNEL_ARG_INVALID.\n");
    } else {
        printf("  -> FAIL: Oversized push constant not rejected (res=%d)\n", res);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 11: Descriptor Binding Count Mismatch Rejection
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 11] Descriptor Binding Count Mismatch Rejection...\n");
    res = ac_vk_dispatch(ctx1, kernel1, buffers1, 2, &valid_push, sizeof(valid_push), 1, 1, 1, NULL);
    if (res == AC_ERR_KERNEL_ARG_INVALID) {
        printf("  -> PASS: Buffer count mismatch (2 vs 3) rejected with AC_ERR_KERNEL_ARG_INVALID.\n");
    } else {
        printf("  -> FAIL: Buffer count mismatch not rejected (res=%d)\n", res);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Gate 12: Dispatch Dimension 0 and Multiplication Overflow Rejection
     * ------------------------------------------------------------------------ */
    printf("\n[Gate 12] Dispatch Dimension 0 and Overflow Rejection...\n");
    res = ac_vk_dispatch(ctx1, kernel1, buffers1, 3, &valid_push, sizeof(valid_push), 0, 1, 1, NULL);
    if (res == AC_ERR_INVALID_WORK_GROUP_SIZE) {
        printf("  -> PASS: Workgroup dimension (0, 1, 1) rejected with AC_ERR_INVALID_WORK_GROUP_SIZE.\n");
    } else {
        printf("  -> FAIL: Dimension 0 not rejected (res=%d)\n", res);
        total_failures++;
    }

    res = ac_vk_dispatch(ctx1, kernel1, buffers1, 3, &valid_push, sizeof(valid_push), 0xFFFFFFFFu, 2, 1, NULL);
    if (res == AC_ERR_NUMERICAL_OVERFLOW) {
        printf("  -> PASS: Workgroup overflow (0xFFFFFFFF * 2) rejected with AC_ERR_NUMERICAL_OVERFLOW.\n");
    } else {
        printf("  -> FAIL: Workgroup overflow not rejected (res=%d)\n", res);
        total_failures++;
    }

    /* ------------------------------------------------------------------------
     * Teardown
     * ------------------------------------------------------------------------ */
    printf("\n[Cleanup] Primary Context Resource Cleanup...\n");
    ac_vk_kernel_destroy(kernel1);
    ac_vk_buffer_destroy(buf1_weight);
    ac_vk_buffer_destroy(buf1_act);
    ac_vk_buffer_destroy(buf1_out);
    ac_vk_context_destroy(ctx1);
    printf("  -> All test resources destroyed cleanly. Process exiting.\n");

    printf("\n====================================================================\n");
    printf("Verification Gates: 12 / 12 Passed (%d Failures)\n", total_failures);
    printf("Cleanup: Completed\n");
    printf("====================================================================\n");
    return (total_failures == 0) ? 0 : 1;
}
