import { ARM_CASES, type ArmCaseId } from "./arm-contract";
import type { Language } from "./language";

const ENGLISH_TITLES: Record<ArmCaseId, string> = {
  "arm-reach-v1": "End-effector reach",
  "arm-pick-place-v1": "Pick and place",
  "arm-relocate-v1": "Obstacle relocation",
  "arm-carry-v1": "Fixed-base carry",
  "arms-handover-v1": "Dual-arm handover",
  "arms-co-carry-v1": "Cooperative tray carry",
};

export function armCaseTitle(id: ArmCaseId, language: Language): string {
  return language === "en" ? ENGLISH_TITLES[id] : ARM_CASES.find((entry) => entry.id === id)?.title ?? id;
}
