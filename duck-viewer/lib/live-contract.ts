import type { DuckFrame } from "./lab";

export function liveContractLabels(ducks: Pick<DuckFrame, "drawing">[] = []) {
  const observations = new Set<number>();
  const actions = new Set<number>();
  for (const duck of ducks) {
    if (!duck.drawing) {
      observations.add(61);
      actions.add(14);
    } else if (duck.drawing.contract === "microduck-drawing-v1") {
      observations.add(83);
      actions.add(15);
    } else if (duck.drawing.contract === "microduck-brush-v2") {
      observations.add(93);
      actions.add(15);
    }
  }
  const label = (values: Set<number>) =>
    values.size ? [...values].sort((left, right) => left - right).join(" / ") : "—";
  return {
    observations: label(observations),
    actions: label(actions),
  };
}
