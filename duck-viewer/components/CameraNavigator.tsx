"use client";

import { useEffect, useState, type MutableRefObject } from "react";

import {
  cameraMotionStart,
  cameraMotionStop,
  requestCameraReset,
  type CameraMotion,
} from "@/lib/camera";
import { loadJSON, saveJSON } from "@/lib/persist";
import type { LabClient } from "@/lib/lab";
import { useLanguage } from "./LanguageProvider";
import { pushToast } from "./Toasts";
import styles from "./CameraNavigator.module.css";

interface CameraNavigatorProps {
  clientRef: MutableRefObject<LabClient | null>;
  connected: boolean;
  defaultOpen?: boolean;
}

interface MotionButtonProps {
  motion: CameraMotion;
  symbol: string;
  label: string;
}

function MotionButton({ motion, symbol, label }: MotionButtonProps) {
  const [active, setActive] = useState(false);

  const stop = () => {
    cameraMotionStop(motion);
    setActive(false);
  };

  useEffect(() => () => cameraMotionStop(motion), [motion]);

  return (
    <button
      type="button"
      className={styles.control}
      aria-label={label}
      aria-pressed={active}
      title={label}
      data-active={active}
      onPointerDown={(event) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        cameraMotionStart(motion);
        setActive(true);
      }}
      onPointerUp={stop}
      onPointerCancel={stop}
      onLostPointerCapture={stop}
      onKeyDown={(event) => {
        if ((event.key === "Enter" || event.key === " ") && !active) {
          event.preventDefault();
          cameraMotionStart(motion);
          setActive(true);
        }
      }}
      onKeyUp={(event) => {
        if (event.key === "Enter" || event.key === " ") stop();
      }}
    >
      {symbol}
    </button>
  );
}

export function CameraNavigator({
  clientRef,
  connected,
  defaultOpen = true,
}: CameraNavigatorProps) {
  const { t } = useLanguage();
  const [open, setOpen] = useState(() =>
    loadJSON("cameraNavigatorOpen", defaultOpen)
  );

  useEffect(() => saveJSON("cameraNavigatorOpen", open), [open]);

  if (!open) {
    return (
      <button
        type="button"
        className={styles.collapsed}
        onClick={() => setOpen(true)}
        aria-label={t("Open camera navigator", "打开相机导航")}
        title={t("Open camera navigator", "打开相机导航")}
      >
        ⌖
      </button>
    );
  }

  return (
    <section className={styles.navigator} aria-label={t("Camera navigator", "相机导航")}>
      <div className={styles.header}>
        <strong>{t("Camera", "相机")}</strong>
        <button
          type="button"
          onClick={() => setOpen(false)}
          aria-label={t("Collapse camera navigator", "收起相机导航")}
          title={t("Collapse camera navigator", "收起相机导航")}
        >
          −
        </button>
      </div>
      <div className={styles.grid}>
        <MotionButton motion="orbitLeft" symbol="↶" label={t("Orbit camera left", "相机向左环绕")} />
        <MotionButton motion="up" symbol="↑" label={t("Raise camera", "升高相机")} />
        <MotionButton motion="orbitRight" symbol="↷" label={t("Orbit camera right", "相机向右环绕")} />
        <MotionButton motion="truckLeft" symbol="←" label={t("Move camera left", "相机向左平移")} />
        <button
          type="button"
          className={`${styles.control} ${styles.home}`}
          onClick={requestCameraReset}
          aria-label={t("Reset camera view", "重置相机视角")}
          title={t("Reset camera view (Shift+R)", "重置相机视角 (Shift+R)")}
        >
          ⌂
        </button>
        <MotionButton motion="truckRight" symbol="→" label={t("Move camera right", "相机向右平移")} />
        <MotionButton motion="dollyOut" symbol="−" label={t("Zoom out", "缩小")} />
        <MotionButton motion="down" symbol="↓" label={t("Lower camera", "降低相机")} />
        <MotionButton motion="dollyIn" symbol="+" label={t("Zoom in", "放大")} />
      </div>
      <button
        type="button"
        className={`${styles.control} ${styles.simReset}`}
        disabled={!connected}
        onClick={() => {
          clientRef.current?.sendReset();
          pushToast(t("Simulation restarted from zero", "仿真已从零重新开始"));
        }}
        aria-label={t("Restart every duck simulation", "重新开始所有小鸭仿真")}
        title={t("Restart every duck simulation (R)", "重新开始所有小鸭仿真 (R)")}
      >
        ↻
      </button>
    </section>
  );
}
