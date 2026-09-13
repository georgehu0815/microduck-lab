import type { Metadata } from "next";
import WingPodPage from "@/components/WingPodPage";

export const metadata: Metadata = {
  title: "WingPod Camera v2 | Microduck Studio",
  description: "WingPod Camera v2 appearance, tennis-task replay, proposed eye housing, BOM, and evidence notes.",
};

export default function Page() {
  return <WingPodPage />;
}
