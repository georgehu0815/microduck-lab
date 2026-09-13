# Microduck arm control implementation

Date: 2026-09-12. Status: **simulation-only control software. Hardware creation
and hardware transport admission are unconditionally denied in this phase. No
motor, arm, serial adapter, or physical E-stop has been connected or validated.
This is not a safety certification.**

## 1. Implemented boundary

`arm_control/` is a Python standard-library control service with one coordinator
owning each configured arm transport. It does not modify or open the original
Microduck body bus.

Fixed identities:

- Original body servos remain IDs `10-14`, `20-24`, and `30-34`; IMU remains ID
  `200`.
- New left arm is IDs `40-45`; new right arm is IDs `50-55`.
- `md-arm-table-v1` is mode `single_arm`, observation/action `66/6`.
- `md-dualarm-table-v1` is mode `dual_arm`, observation/action `116/12`.
- Each arm action is five normalized joint velocities mapped to at most
  `0.4 rad/s`, then one normalized gripper-width velocity mapped to at most
  `0.01 m/s`.
- The coordinator runs at 50 Hz. It applies per-joint acceleration, position,
  finite-value, sequence, deadline, lease, device, mode, contract, and joint-map
  checks before the transport sees a target.

The six-joint order is:

```text
base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, gripper_width
```

The RPC server is JSON-RPC 2.0, one JSON object per newline, over a Unix socket.
Unknown envelope and parameter fields are rejected. `manipulation.capabilities`
is generated from immutable local contracts; no RPC or deployment file can add
physical capabilities.

This Unix coordinator is separate from `scripts/arm_lab.py`, which is a
MuJoCo-backed localhost HTTP simulation service on `127.0.0.1:8812`. The HTTP
service cannot acquire a coordinator lease, open a transport, or command motors.

## 2. Safety state and ownership behavior

States are `LOCKED`, `READY`, `EXECUTING`, `PROTECTIVE_STOP`, and `ESTOP`.

- The default daemon transport is `locked`; it cannot acquire or move an arm.
- `dynamixel` daemon startup, the public serial factory, direct admission of a
  transport marked as hardware, and the legacy `hardware_authorized=True`
  constructor path all fail before a device can be opened or controlled.
- A lease owns its listed arms exclusively and atomically. A dual-arm lease
  cannot be partially acquired.
- Commands pin `mode`, `contract_id`, dimensions, device set, session epoch,
  monotonic sequence, monotonic deadline, and each joint-map hash.
- The five-control-period watchdog starts when the lease is acquired, not when
  the first command arrives. More than 100 ms without an accepted command causes
  a latched protective stop. The lease ceiling is 500 ms.
- The body ownership adapter is checked on every control tick before any arm
  write. Loss or failure stops every arm in that lease and removes its authority.
- A late control tick, transport error, invalid measured state, lost body
  interlock, expired command, or expired lease also causes a protective stop.
- An unexpected ticker exception protective-stops the simulated coordinator,
  removes leases, latches the coordinator unavailable to reset or reacquisition,
  and shuts down the RPC server instead of leaving commands available without
  the 50 Hz control loop.
- Normal daemon shutdown also protective-stops active simulated transports,
  removes leases, releases body ownership, and closes transports.
- `manipulation.stop` is idempotent in effect and holds the last position; it
  does not automatically open the gripper.
- `manipulation.estop` invokes the configured non-hardware transport's emergency
  stop behavior and latches `ESTOP`. Reset requires the exact acknowledgement
  `I inspected the arm and cleared the cause`, a true
  `physical_estop_released` claim, and transport confirmation. This is simulation
  state-machine coverage only; no physical E-stop adapter exists.
- No stop automatically resumes. A stopped arm must be manually reset and a
  new lease acquired.

These are software containment mechanisms. They do not replace a correctly
rated hardware E-stop, power contactor, current protection, guarding, or a
supervised bench procedure.

## 3. Body movement interlock

Arm control never writes body joints. It depends on the `BodyInterlock` adapter
interface. Tests use `SimulatedBodyInterlock`, which provides exclusive
ownership and proves that acquisition/maintenance/release are enforced by the
coordinator.

`RobotdStopAdapter` sends the existing discrete `robot.stop` JSON-RPC request.
It deliberately reports `exclusive = false`: current `robotd` can stop body
velocity, but it cannot grant an exclusive movement lease, so another client
could submit a later body movement intent. Coordinator acquisition therefore
fails closed with this adapter, independently of the hardware hard-deny. A
future robotd integration must add a real, expiring body-hold lease before a
separately reviewed hardware phase can be considered.

This is the explicit integration point; there is no direct body joint write or
second body-bus owner in this package.

## 4. Run the locked or simulated daemon

From the repository root:

```bash
# Default: serves capabilities/state but cannot acquire an arm.
python3 -m arm_control.daemon \
  --socket /tmp/microduck-arm-locked.sock

# Explicit coordinator simulation: runs both simulated arms at 50 Hz.
python3 -m arm_control.daemon \
  --transport simulation \
  --socket /tmp/microduck-arm.sock
```

In another shell, exercise a single-arm lease and command:

```bash
python3 arm_control/examples/unix_rpc_client.py /tmp/microduck-arm.sock
```

Raw capability request:

```bash
python3 - <<'PY'
import json, socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/tmp/microduck-arm.sock")
s.sendall(b'{"jsonrpc":"2.0","id":1,"method":"manipulation.capabilities"}\n')
print(json.loads(s.makefile("rb").readline()))
PY
```

The mutating methods are:

```text
manipulation.acquire
manipulation.command
manipulation.release
manipulation.stop
manipulation.estop
manipulation.reset
```

Read-only methods are `manipulation.capabilities` and `manipulation.state`.
The example client shows the full acquire and command envelopes, including the
server-issued session epoch and joint-map hash.

`SimulatedTransport` is a deterministic coordinator test double, not a hardware
backend and not the MuJoCo HTTP simulator. Its current, temperature, and voltage
samples are synthetic: current is fixed at `0 mA`, temperature at `25 C`, and
voltage at `5 V`. These values do not represent an XL330, a power supply, thermal
state, payload, or any physical measurement.

## 5. Hardware is hard-denied

There is no hardware permit implementation in this phase. The following all
fail before opening a device:

- `arm_control.daemon --transport dynamixel`;
- `open_xl330_transport(...)`;
- `PosixSerialStream.open(...)`;
- constructing a `Coordinator` with any transport whose `hardware` attribute is
  true;
- passing the removed `hardware_authorized=True` constructor keyword.

`load_hardware_deployment(...)` remains only as an offline parser and validator
for a possible future manifest format. A valid manifest or unexpired bench file
does not authorize hardware and is never consumed by the daemon. In particular,
its expiration check is not a continuously maintained permit and must not be
treated as one.

`arm_control.dynamixel` implements Protocol 2 CRC, byte stuffing, read/write,
sync-read, sync-write, status validation, XL330 goal-position packet generation,
and present current/velocity/position/voltage/temperature decoding as offline
preparation. POSIX serial construction and opening are disabled. The code has
only been packet-tested against simulated byte data. No U2D2, XL330, UART, USB
device, or motor was opened during this work.

Hardware remains blocked until all of these independently reviewed facilities
exist:

1. Startup discovery verifies the exact servo model and IDs on each isolated arm
   bus before torque can be enabled.
2. Torque mode, gains, current limits, direction, zero position, travel limits,
   voltage, and thermal limits are commissioned from measured hardware.
3. A rated physical E-stop and power-removal path provide independently readable
   state and verified reset behavior.
4. Robotd provides an exclusive, expiring body movement-hold lease that is
   renewed every arm control tick.
5. Authorization validity and expiration are continuously enforced rather than
   checked only at process startup.
6. An independent watchdog outside this Python process removes arm authority or
   power if the process, scheduler, or host stalls.

## 6. Proposed systemd deployment

- `arm_control/systemd/microduck-arm-control.service` is the installable default:
  dedicated user/group, `0660` socket under a private runtime directory,
  `PrivateDevices=yes`, and locked transport.
- `arm_control/systemd/microduck-arm-control-bench.service` is a non-enabled,
  hard-denied placeholder. It runs as the unprivileged service user, retains
  `PrivateDevices=yes`, grants no device paths, and invokes a daemon mode that
  always refuses startup. It is not a usable bench deployment.

## 7. Verification and remaining hardware work

Run:

```bash
python3 -m unittest discover -s arm_control/tests -v
python3 -m compileall -q arm_control
```

Covered in software tests:

- fixed body/arm ID separation and stable joint-map hashes;
- offline deployment/authorization document checks;
- per-arm exclusive leases and dual/single contract shape;
- mode, device, epoch, sequence, deadline, joint-map, finite, and normalized
  action rejection;
- 50 Hz velocity/acceleration integration and position limiting;
- command watchdog, lease stop, protective stop, ESTOP, and manual reset;
- body-interlock acquisition and refusal of the non-exclusive robotd adapter;
- body ownership maintenance before every write and full-lease stop propagation;
- watchdog timing beginning at acquisition;
- ticker-exception safe-stop, lease removal, and RPC shutdown;
- unconditional daemon, coordinator, and serial-factory hardware denial;
- real Unix socket framing/permissions and strict JSON-RPC fields;
- Protocol 2 ping CRC, read/write encoding, sync-write encoding, and status CRC.

Not tested and not claimed:

- serial timing, half-duplex direction behavior, U2D2 compatibility, device
  return ordering, XL330 register/model identity, or EEPROM settings;
- actual joint directions, zero counts, gripper geometry, current units, safe
  torque/gain, thermal/voltage thresholds, collision bounds, or payload;
- physical E-stop feedback, power-cut behavior, dropped-load behavior, or body
  movement exclusion on a real Microduck;
- startup servo identity checks, torque-enable sequencing, continuously enforced
  authorization expiry, or an independent hardware watchdog;
- real-time scheduling guarantees on the Microduck host.

Those items remain prerequisites for a separately reviewed hardware phase.
Passing these unit tests, supplying capability files, or changing a boolean does
not authorize hardware motion.
