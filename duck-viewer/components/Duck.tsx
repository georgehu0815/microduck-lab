"use client";

// One duck: a Three.js group per MuJoCo body, poses smoothed toward the
// latest streamed frame. All geoms of a body are merged into ONE geometry at
// load time, so a duck is ~16 draw calls — the naive per-geom version
// (70 meshes/duck, all casting shadows) lost the WebGL context with 8 ducks
// on screen. Per-geom MJCF material colors survive the merge as a
// vertex-color channel, so the eye ring / mouth / shells keep their own
// colors without costing extra draw calls.

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import type {
  DrawingPayload,
  DuckFrame,
  Scene,
  SceneGeom,
  ScenePrimitive,
  SceneTendon,
} from "@/lib/lab";
import { assignDrag } from "@/lib/assign";
import { captureWantsCleanFrame } from "@/lib/record";
import { getSelectedDuck } from "@/lib/select";
import { getDuckLabels } from "@/lib/ui";

// FALLBACK body-name → color, used only against servers that predate rgba
// streaming (whole body painted one guessed color).
function bodyColor(name: string): string {
  if (/foot|ankle/.test(name)) return "#e8862e"; // webbed orange
  if (/head|beak/.test(name)) return "#f5efe0";
  if (/neck/.test(name)) return "#e8862e";
  if (/trunk/.test(name)) return "#f5efe0"; // cream body
  if (/hip|knee|leg/.test(name)) return "#4a4e57"; // dark joints
  return "#c8c2b4";
}

// The MJCF materials are OnShape-export appearances, and a few don't match
// the printed robot: the yellow eye ring exports dark grey, the soft TPU
// mouth pink, and the beak/shoes a flat gold where the real parts are
// orange with yellow soles. Override those by material name; everything
// else renders straight from the streamed rgba.
const MATERIAL_FIX: Record<string, string> = {
  basketball_mat: "#d87519", // texture is not streamed; keep the ball recognizably orange
  face_part_material: "#a5a6a2", // face panel — light grey, exports dark slate
  noenoeil_material: "#f2b705", // eye ring — yellow print, not grey
  soft_mouth_top_material: "#e06a1e", // soft TPU mouth seam
  jaw_soft_material: "#e06a1e",
  jaw_material: "#e8862e", // lower beak: orange, not gold
  bottom_head_shell_material: "#e8862e",
  foot_left_material: "#e8862e", // TPU shoes orange…
  foot_right_material: "#e8862e",
  ankle_left_material: "#e8862e",
  ankle_right_material: "#e8862e",
  sole_left_material: "#f2b705", // …with yellow soles
  sole_right_material: "#f2b705",
};

/** Resolved sRGB→linear color for one geom (override → MJCF rgba → fallback). */
function geomColor(g: SceneGeom, bodyName: string, out: THREE.Color): THREE.Color {
  const fix = g.mat ? MATERIAL_FIX[g.mat] : undefined;
  if (fix) return out.set(fix);
  if (g.rgba) return out.setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);
  return out.set(bodyColor(bodyName));
}

export interface BodyGeometry {
  name: string;
  geometry: THREE.BufferGeometry | null;
}

function primitiveColor(
  primitive: ScenePrimitive,
  bodyName: string,
  out: THREE.Color
): THREE.Color {
  const fix = primitive.mat ? MATERIAL_FIX[primitive.mat] : undefined;
  if (fix) return out.set(fix);
  if (primitive.rgba.length >= 3) {
    return out.setRGB(
      primitive.rgba[0],
      primitive.rgba[1],
      primitive.rgba[2],
      THREE.SRGBColorSpace
    );
  }
  return out.set(bodyColor(bodyName));
}

function paintGeometry(geometry: THREE.BufferGeometry, color: THREE.Color) {
  const count = geometry.getAttribute("position").count;
  const colors = new Float32Array(count * 3);
  for (let index = 0; index < count; index++) color.toArray(colors, index * 3);
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
}

function primitiveGeometry(primitive: ScenePrimitive): THREE.BufferGeometry {
  const [sizeX = 0, sizeY = sizeX, sizeZ = sizeY] = primitive.size;
  let geometry: THREE.BufferGeometry;
  if (primitive.type === "box") {
    geometry = new THREE.BoxGeometry(sizeX * 2, sizeY * 2, sizeZ * 2);
  } else if (primitive.type === "sphere") {
    geometry = new THREE.SphereGeometry(sizeX, 12, 8);
  } else if (primitive.type === "ellipsoid") {
    geometry = new THREE.SphereGeometry(1, 12, 8);
    geometry.scale(sizeX, sizeY, sizeZ);
  } else if (primitive.type === "cylinder") {
    geometry = new THREE.CylinderGeometry(sizeX, sizeX, sizeY * 2, 12);
    geometry.rotateX(Math.PI / 2);
  } else {
    geometry = new THREE.CapsuleGeometry(sizeX, sizeY * 2, 6, 12);
    geometry.rotateX(Math.PI / 2);
  }
  geometry.deleteAttribute("normal");
  geometry.deleteAttribute("uv");
  return geometry;
}

/** Merge every geom of every body into one geometry per body (body-local
 *  frame), painting each geom's material color into a vertex-color channel. */
export function buildBodyGeometries(scene: Scene): BodyGeometry[] {
  const meshGeos = scene.meshes.map((m) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(m.v, 3));
    g.setIndex(m.f);
    return g;
  });
  const mat = new THREE.Matrix4();
  const quat = new THREE.Quaternion();
  const col = new THREE.Color();
  const out = scene.bodies.map((name, b) => {
    const meshParts = scene.geoms
      .filter((g) => g.body === b)
      .map((g) => {
        const geo = meshGeos[g.mesh].clone();
        quat.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]); // wxyz → xyzw
        mat.compose(new THREE.Vector3(...g.pos), quat, new THREE.Vector3(1, 1, 1));
        geo.applyMatrix4(mat);
        geomColor(g, name, col);
        paintGeometry(geo, col);
        return geo;
      });
    const primitiveParts = (scene.primitives ?? [])
      .filter((primitive) => primitive.body === b)
      .map((primitive) => {
        const geometry = primitiveGeometry(primitive);
        quat.set(
          primitive.quat[1] ?? 0,
          primitive.quat[2] ?? 0,
          primitive.quat[3] ?? 0,
          primitive.quat[0] ?? 1
        );
        mat.compose(
          new THREE.Vector3(
            primitive.pos[0] ?? 0,
            primitive.pos[1] ?? 0,
            primitive.pos[2] ?? 0
          ),
          quat,
          new THREE.Vector3(1, 1, 1)
        );
        geometry.applyMatrix4(mat);
        primitiveColor(primitive, name, col);
        paintGeometry(geometry, col);
        return geometry;
      });
    const parts = [...meshParts, ...primitiveParts];
    if (!parts.length) return { name, geometry: null };
    const merged = mergeGeometries(parts, false);
    parts.forEach((p) => p.dispose());
    merged.computeVertexNormals();
    return { name, geometry: merged };
  });
  meshGeos.forEach((g) => g.dispose());
  return out;
}

function Tendons({
  tendons,
  bodyRefs,
}: {
  tendons: SceneTendon[];
  bodyRefs: React.MutableRefObject<(THREE.Group | null)[]>;
}) {
  const lineRefs = useRef<(THREE.LineSegments | null)[]>([]);
  const point = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    tendons.forEach((tendon, tendonIndex) => {
      const line = lineRefs.current[tendonIndex];
      const positions = line?.geometry.getAttribute("position");
      if (!line || !(positions instanceof THREE.BufferAttribute)) return;
      let positionIndex = 0;
      for (let siteIndex = 0; siteIndex < tendon.sites.length - 1; siteIndex++) {
        for (const site of [tendon.sites[siteIndex], tendon.sites[siteIndex + 1]]) {
          const body = bodyRefs.current[site.body];
          if (body) {
            point
              .set(site.pos[0] ?? 0, site.pos[1] ?? 0, site.pos[2] ?? 0)
              .applyQuaternion(body.quaternion)
              .add(body.position);
          } else {
            point.set(0, 0, 0);
          }
          positions.setXYZ(positionIndex, point.x, point.y, point.z);
          positionIndex += 1;
        }
      }
      positions.needsUpdate = true;
    });
  });

  return tendons.map((tendon, tendonIndex) => {
    const segmentCount = Math.max(0, tendon.sites.length - 1);
    const color = tendon.rgba.length >= 3
      ? new THREE.Color().setRGB(
          tendon.rgba[0],
          tendon.rgba[1],
          tendon.rgba[2],
          THREE.SRGBColorSpace
        )
      : new THREE.Color("#b9c1cc");
    return (
      <lineSegments
        key={tendonIndex}
        ref={(element) => void (lineRefs.current[tendonIndex] = element)}
        frustumCulled={false}
      >
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array(segmentCount * 6), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={color}
          linewidth={tendon.width}
          transparent={(tendon.rgba[3] ?? 1) < 1}
          opacity={tendon.rgba[3] ?? 1}
          depthWrite={(tendon.rgba[3] ?? 1) >= 1}
        />
      </lineSegments>
    );
  });
}

const MAX_DRAWING_POINTS = 6000;
const MAX_DRAWING_SEGMENT_VALUES = (MAX_DRAWING_POINTS - 1) * 2 * 3;
const srgbToLinear = (channel: number) =>
  channel <= 0.04045
    ? channel / 12.92
    : ((channel + 0.055) / 1.055) ** 2.4;
const PENCIL_COLOR = [37, 39, 41].map((channel) =>
  srgbToLinear(channel / 255)
);

export function writeDrawingSegments(
  drawing: DrawingPayload | null | undefined,
  target: Float32Array,
  colorTarget?: Float32Array
): number {
  if (!drawing || drawing.assistance !== 0 || !Array.isArray(drawing.points)) {
    return 0;
  }
  const points = drawing.points.slice(-MAX_DRAWING_POINTS);
  const isBrush = drawing.contract === "microduck-brush-v2";
  if (isBrush && (!Array.isArray(drawing.colors) || !Array.isArray(drawing.palette) ||
    drawing.colors.length !== drawing.points.length || drawing.palette.some(color => !Array.isArray(color)))) {
    return 0;
  }
  const pointColors = isBrush
    ? drawing.colors.slice(-MAX_DRAWING_POINTS)
    : undefined;
  const palette = isBrush
    ? drawing.palette.map((color) => color.map(srgbToLinear))
    : undefined;
  if (
    drawing.contract !== "microduck-drawing-v1" &&
    drawing.contract !== "microduck-brush-v2"
  ) {
    return 0;
  }
  if (
    isBrush &&
    (pointColors?.length !== points.length ||
      drawing.palette.length !== 4 ||
      drawing.palette.some(
        (color) =>
          color.length !== 3 ||
          color.some((channel) => !Number.isFinite(channel))
      ) ||
      !Number.isFinite(drawing.brush_radius) ||
      drawing.brush_radius <= 0)
  ) {
    return 0;
  }
  let valueIndex = 0;
  for (let index = 1; index < points.length; index++) {
    const previous = points[index - 1];
    const current = points[index];
    if (previous[3] !== current[3]) continue;
    let color: readonly number[] = PENCIL_COLOR;
    if (drawing.contract === "microduck-brush-v2") {
      const colorIndex = pointColors![index]!;
      if (
        colorIndex !== pointColors![index - 1] ||
        !Number.isInteger(colorIndex) ||
        colorIndex < 0 ||
        colorIndex >= drawing.palette.length
      ) {
        continue;
      }
      color = palette![colorIndex]!;
    }
    for (const point of [previous, current]) {
      target[valueIndex++] = point[0];
      target[valueIndex++] = point[1];
      target[valueIndex++] = point[2];
      if (colorTarget) {
        colorTarget[valueIndex - 3] = color[0];
        colorTarget[valueIndex - 2] = color[1];
        colorTarget[valueIndex - 1] = color[2];
      }
    }
  }
  return valueIndex / 3;
}

function DrawingInk({
  frameRef,
}: {
  frameRef: React.MutableRefObject<DuckFrame | null>;
}) {
  const lineRef = useRef<THREE.LineSegments>(null);
  const positions = useMemo(
    () => new Float32Array(MAX_DRAWING_SEGMENT_VALUES),
    []
  );
  const colors = useMemo(
    () => new Float32Array(MAX_DRAWING_SEGMENT_VALUES),
    []
  );

  useFrame(() => {
    const line = lineRef.current;
    if (!line) return;
    const count = writeDrawingSegments(
      frameRef.current?.drawing,
      positions,
      colors
    );
    line.geometry.setDrawRange(0, count);
    for (const name of ["position", "color"]) {
      const attribute = line.geometry.getAttribute(name);
      if (attribute instanceof THREE.BufferAttribute) attribute.needsUpdate = true;
    }
  });

  return (
    <lineSegments ref={lineRef} frustumCulled={false} renderOrder={4}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <lineBasicMaterial
        vertexColors
        depthWrite={false}
        toneMapped={false}
      />
    </lineSegments>
  );
}

export function Duck({
  bodies,
  tendons = [],
  frameRef,
  offset,
  label,
  duckId,
}: {
  bodies: BodyGeometry[];
  tendons?: SceneTendon[];
  frameRef: React.MutableRefObject<DuckFrame | null>;
  offset: [number, number]; // grid offset in MuJoCo XY
  label: string;
  duckId: string; // stable stream id ("d0"…, "trainee") — assignment target
}) {
  const bodyRefs = useRef<(THREE.Group | null)[]>([]);
  const labelRef = useRef<THREE.Group>(null);
  const labelDivRef = useRef<HTMLDivElement>(null);
  const spawnDivRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const tmpP = useMemo(() => new THREE.Vector3(), []);
  const tmpQ = useMemo(() => new THREE.Quaternion(), []);

  useEffect(() => {
    bodyRefs.current.length = bodies.length;
  }, [bodies.length]);

  useFrame((_, dt) => {
    const duck = frameRef.current;
    if (!duck) return;
    const alpha = 1 - Math.exp(-16 * Math.min(dt, 0.1));
    duck.bodies.forEach((pose, b) => {
      const grp = bodyRefs.current[b];
      if (!grp) return;
      tmpP.set(pose[0], pose[1], pose[2]);
      tmpQ.set(pose[4], pose[5], pose[6], pose[3]); // wxyz → xyzw
      grp.position.lerp(tmpP, alpha);
      grp.quaternion.slerp(tmpQ, alpha);
    });
    // Float the label above the trunk (body 1 = trunk_base in this model).
    const trunk = duck.bodies[1];
    if (labelRef.current && trunk) {
      tmpP.set(trunk[0], trunk[1], trunk[2] + 0.22);
      labelRef.current.position.lerp(tmpP, alpha);
    }
    // Drop-target highlight while a policy chip is dragged/armed: floor ring
    // under the duck + emphasized label. Driven straight off the shared store
    // (no React state — this flips at pointer speed). The same ring doubles
    // as the click-selection marker (Delete removes the selected duck); an
    // active assign hover wins the color so the drop target stays legible.
    const hovered = assignDrag.mode !== null && assignDrag.hoverDuck === duckId;
    const selected = getSelectedDuck() === duckId;
    // The ring lives IN the 3D scene, so unlike the DOM labels it would land
    // in captured footage — hide it while a 🎥 take is framing/rolling.
    // (📷 snapshots hide it themselves via the hideInCapture tag below.)
    const filming = captureWantsCleanFrame();
    if (ringRef.current) {
      ringRef.current.visible = (hovered || selected) && !filming;
      (ringRef.current.material as THREE.MeshBasicMaterial).color.set(
        hovered ? "#7db8d8" : "#e8b24a"
      );
      if (trunk) {
        tmpP.set(trunk[0], trunk[1], 0.004);
        ringRef.current.position.lerp(tmpP, alpha);
      }
    }
    if (labelDivRef.current) {
      const s = labelDivRef.current.style;
      // HUD 🏷 toggle, applied per-frame like the rest of the label styling
      // (a React subscription inside the Canvas tree flushed a beat late).
      s.display = getDuckLabels() ? "" : "none";
      s.transform = hovered ? "scale(1.3)" : selected ? "scale(1.15)" : "none";
      s.color = hovered ? "#9fd4f0" : selected ? "#e8b24a" : "#fff";
      s.fontWeight = hovered || selected ? "700" : "400";
    }
    // Spawn note updates straight off the stream (textContent, no React
    // churn) — it changes at every episode reset.
    if (spawnDivRef.current) {
      const spawn = duck.spawn && duck.spawn !== "standing" ? `↻ ${duck.spawn}` : "";
      const parts = [spawn];
      if (duck.assist) parts.push("🤝 spotting");
      if (duck.handed && duck.handoff) parts.push(`→ ${duck.handoff}`);
      const txt = parts.filter(Boolean).join(" · ");
      if (spawnDivRef.current.textContent !== txt)
        spawnDivRef.current.textContent = txt;
    }
  });

  return (
    <group position={[offset[0], offset[1], 0]}>
      {bodies.map((body, b) =>
        body.geometry ? (
          <group key={b} ref={(el) => void (bodyRefs.current[b] = el)}>
            <mesh geometry={body.geometry}>
              <meshStandardMaterial vertexColors roughness={0.55} metalness={0.08} />
            </mesh>
          </group>
        ) : (
          <group key={b} ref={(el) => void (bodyRefs.current[b] = el)} />
        )
      )}
      {tendons.length > 0 && <Tendons tendons={tendons} bodyRefs={bodyRefs} />}
      <DrawingInk frameRef={frameRef} />
      {/* drop-target ring, flat on the floor (XY plane in this Z-up group).
          hideInCapture: 📷 snapshots hide it for their capture render. */}
      <mesh
        ref={ringRef}
        visible={false}
        position={[0, 0, 0.004]}
        userData={{ hideInCapture: true }}
      >
        <ringGeometry args={[0.16, 0.19, 48]} />
        <meshBasicMaterial
          color="#7db8d8"
          transparent
          opacity={0.85}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <group ref={labelRef}>
        {/* DOM label (drei Text's GPU glyph atlas was losing the WebGL
            context in the embedded browser — keep labels off the GPU).
            zIndexRange tops out below the overlay panels (zIndex 20) so
            labels can never scribble over the HUD/policies/teach UI.
            Stays mounted when labels are toggled off — the useFrame above
            flips the inner div's display instead. */}
        <Html center zIndexRange={[10, 0]} style={{ pointerEvents: "none" }}>
          <div
            ref={labelDivRef}
            data-duck-id={duckId}
            style={{
              color: "#fff",
              fontFamily: "ui-monospace, Menlo, monospace",
              fontSize: 11,
              whiteSpace: "nowrap",
              textShadow: "0 1px 3px rgba(0,0,0,0.9)",
              opacity: 0.9,
              transition: "transform 120ms ease, color 120ms ease",
            }}
          >
            {label}
            <div
              ref={spawnDivRef}
              style={{
                fontSize: 9,
                color: "#e8b24a",
                textAlign: "center",
                minHeight: 11,
              }}
            />
          </div>
        </Html>
      </group>
    </group>
  );
}
