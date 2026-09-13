import type { Policy } from "./lab";

export type RobotCategory = "microduck" | "wingpod" | "humanoid";

export const ROBOT_CATEGORIES: readonly RobotCategory[] = [
  "microduck",
  "wingpod",
  "humanoid",
] as const;

export function policyRobotCategory(policy: Policy): RobotCategory {
  if (policy.category && ROBOT_CATEGORIES.includes(policy.category)) return policy.category;
  const identity = `${policy.id} ${policy.label} ${policy.recipe ?? ""}`.toLowerCase();
  if (identity.includes("humanoid")) return "humanoid";
  if (identity.includes("wingpod")) return "wingpod";
  return "microduck";
}

export function policySupportsCategory(policy: Policy, category: RobotCategory): boolean {
  const policyCategory = policyRobotCategory(policy);
  if (category === "wingpod") {
    return policyCategory === "microduck" || policyCategory === "wingpod";
  }
  return policyCategory === category;
}

export function policySkillKey(policy: Policy): string {
  if (policy.group === "pollen") return policy.id;
  if (policy.group === "studio") {
    return `studio:${policy.recipe ?? policy.id.split(":", 2)[1]?.split("/", 1)[0] ?? policy.label}`;
  }
  const runName = policy.id.replace(/^(?:run|ckpt):/, "").split("@", 1)[0];
  const chainName = policy.chain ?? runName.replace(/-s\d+$/, "");
  const teachSkill = chainName.match(/^teach-(.+?)-[a-f0-9]{6,}$/i)?.[1];
  return teachSkill ? `teach:${teachSkill}` : `run:${chainName}`;
}

export function newestPoliciesPerSkill(policies: readonly Policy[]): Policy[] {
  const newest = new Map<string, { policy: Policy; index: number }>();
  policies.forEach((policy, index) => {
    const key = policySkillKey(policy);
    const current = newest.get(key);
    if (!current || (policy.mtime ?? 0) > (current.policy.mtime ?? 0)) {
      newest.set(key, { policy, index });
    }
  });
  return [...newest.values()]
    .sort((left, right) => left.index - right.index)
    .map(({ policy }) => policy);
}