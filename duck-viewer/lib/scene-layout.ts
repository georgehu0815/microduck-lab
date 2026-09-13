export interface SceneLayoutItem {
  sceneKey?: string;
}

export function gridOffsets(
  items: number | readonly SceneLayoutItem[],
  standardSpacing = 0.65,
  customSceneSpacing = 2.6
): [number, number][] {
  const count = typeof items === "number" ? items : items.length;
  const spacing =
    typeof items !== "number" && items.some((item) => item.sceneKey)
      ? customSceneSpacing
      : standardSpacing;
  const cols = Math.ceil(Math.sqrt(count));
  return Array.from({ length: count }, (_, index) => [
    (index % cols) * spacing -
      ((Math.min(count, cols) - 1) * spacing) / 2,
    Math.floor(index / cols) * spacing,
  ]);
}
